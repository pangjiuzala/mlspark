import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.jblas.DoubleMatrix
import scala.math.Ordering
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.evaluation.RankingMetrics

object ALSTest {
  def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double =
    {
      vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
    }
  /* compute the average precision at K*/
  def avgPrecsionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double =
    {
      val predK = predicted.take(k)
      var score = 0.0
      var numHits = 0.0
      for ((p, i) <- predK.zipWithIndex) {
        if (actual.contains(p)) {
          numHits += 1.0
          score += numHits / (i.toDouble + 1.0)
        }
      }
      if (actual.isEmpty) {
        1.0
      } else {
        score / math.min(actual.size, k).toDouble
      }

    }
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ALS Application")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("hdfs://master:9000/user/root/input/ml/u.data")
    print(rawData.first())
    val rawRating = rawData.map(_.split("\t").take(3))
    val ratings = rawRating.map {
      case Array(user, movie, rating) => Rating(
        user.toInt, movie.toInt, rating.toFloat)
    }
    print(ratings.first())
    /*We'll use rank of 50, 10 iterations, and a lambda parameter of 0.01 to illustrate how
to train our model:*/
    val model = ALS.train(ratings, 50, 10, 0.01)
    print(model.userFeatures.count)
    val predictedRating = model.predict(789, 123)
    print(predictedRating)
    val userId = 789
    val K = 10
    val topKRecs = model.recommendProducts(userId, K)
    println(topKRecs.mkString("\n"))
    val movies = sc.textFile("hdfs://master:9000/user/root/input/ml/u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt,
      array(1))).collectAsMap()
    println(titles(123))
    val moviesForUser = ratings.keyBy(_.user).lookup(789)
    println(moviesForUser.size)
    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.
      product), rating.rating)).foreach(println)
    topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)
    val itemId = 567
    val itemFactor = model.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)
    println(cosineSimilarity(itemVector, itemVector))
    val sims = model.productFeatures.map {
      case (id, factor) =>
        val factorVector = new DoubleMatrix(factor)
        val sim = cosineSimilarity(factorVector, itemVector)
        (id, sim)
    }
    val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double] {
      case (id, similarity) => similarity
    })
    println(sortedSims.take(10).mkString("\n"))
    /*   check our item-to-item similarity we will take the numbers 1 to 11 in the list:*/
    val sortedSims2 = sims.top(K + 1)(Ordering.by[(Int, Double), Double] {
      case (id, similarity) => similarity
    })
    println(sortedSims2.slice(1, 11).map {
      case (id, sim) => (titles(id), sim)
    }.mkString("\n"))
    /*Mean Squared Error*/
    val actualRating = moviesForUser.take(1)(0)
    val predictRating = model.predict(789, actualRating.product)
    val squaredError = math.pow(predictedRating - actualRating.rating, 2.0)
    println("actualRating is " + actualRating + "\n" + "predictRating is " + predictedRating + "\n" +
      "squaredError is " + squaredError)
    val usersProducts = ratings.map {
      case Rating(user, product, rating) => (user, product)
    }
    val predictions = model.predict(usersProducts).map {
      case Rating(user, product, rating) => ((user, product), rating)
    }
    val ratingsAndPredictions = ratings.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)
    val MSE = ratingsAndPredictions.map {
      case ((user, product), (actual, predicted)) => math.pow((actual -
        predicted), 2)
    }.reduce(_ + _) / ratingsAndPredictions.count
    println("MSE= " + MSE)
    val RMSE = math.sqrt(MSE)
    println("Root MSE= " + RMSE)
    /* compute the APK metric*/
    val actualMovies = moviesForUser.map(_.product)
    val predictedMovies = topKRecs.map(_.product)
    val apk10 = avgPrecsionK(actualMovies, predictedMovies, 10)
    println(apk10)
    /*collect the item factors and form a DoubleMatrix object from them:*/
    val itemFactors = model.productFeatures.map { case (id, factor) => factor }.collect()
    val itemMatrix = new DoubleMatrix(itemFactors)
    println(itemMatrix.rows, itemMatrix.columns)
    /*  distribute the item matrix as a broadcast variable so that it is available on each worker node*/
    val imBroadcast = sc.broadcast(itemMatrix)
    /* sort*/
    val allRecs = model.userFeatures.map {
      case (userId, array) =>
        val userVector = new DoubleMatrix(array)
        val scores = imBroadcast.value.mmul(userVector)
        val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
        val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
        (userId, recommendedIds)
    }
    val userMovies = ratings.map {
      case Rating(user, product, rating) =>
        (user, product)
    }.groupBy(_._1)
    val MAPK = allRecs.join(userMovies).map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecsionK(actual, predicted, K)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision at K= " + MAPK)
    /*Compute RMSE and MSE*/
    val predictedAndTrue = ratingsAndPredictions.map {
      case ((uesr, product), (predicted, actual)) => (predicted, actual)
    }
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)
    println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error = " + regressionMetrics.
      rootMeanSquaredError)
    /*  Mean Average Precision */
    val predictedAndTrueForRanking = allRecs.join(userMovies).map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2)
        (predicted.toArray, actual.toArray)
    }
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
    println("Mean Average Precision = " + rankingMetrics.meanAveragePrecision)
    val MAPK2000 = allRecs.join(userMovies).map {
      case (userId, (predicted,
        actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecsionK(actual, predicted, 2000)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision = " + MAPK2000)
  }

}