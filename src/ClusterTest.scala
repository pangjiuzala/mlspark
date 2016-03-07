

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.ml.clustering.KMeans
import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.numerics.pow
import org.apache.spark.mllib.clustering.KMeans
object ClusterTest {
  def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) = pow(v1 - v2, 2).sum
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("cluster")
    val sc = new SparkContext(conf)
    val movies = sc.textFile("hdfs://master:9000/user/root/input/ml/u.item")
    //println(movies.first())
    val genres = sc.textFile("hdfs://master:9000/user/root/input/ml/u.genre")
    //genres.take(5).foreach(println)
    /* Extracting movie genre labels*/
    val genreMap = genres.filter(!_.isEmpty).map(line => line.
      split("\\|")).map(array => (array(1), array(0))).collectAsMap
    /*The preceding code will provide the following output:*/
    //println(genreMap)
    val titleAndGenres = movies.map(_.split("\\|")).map { array =>
      val genres = array.toSeq.slice(5, array.size)
      val genresAssigned = genres.zipWithIndex.filter {
        case (g, idx) =>
          g == "1"
      }.map {
        case (g, idx) =>
          genreMap(idx.toString)
      }
      (array(0).toInt, (array(1), genresAssigned))
    }
    //println(titleAndGenres.first)

    val rawData = sc.textFile("hdfs://master:9000/user/root/input/ml/u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map {
      case Array(user, movie, rating) =>
        Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    ratings.cache
    val alsModel = ALS.train(ratings, 50, 10, 0.1)
    val movieFactors = alsModel.productFeatures.map {
      case (id, factor) =>
        (id, Vectors.dense(factor))
    }
    val movieVectors = movieFactors.map(_._2)
    val userFactors = alsModel.userFeatures.map {
      case (id, factor) =>
        (id, Vectors.dense(factor))
    }
    val userVectors = userFactors.map(_._2)
    val movieMatrix = new RowMatrix(movieVectors)
    val movieMatrixSummary =
      movieMatrix.computeColumnSummaryStatistics()
    val userMatrix = new RowMatrix(userVectors)
    val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
    //println("Movie factors mean: " + movieMatrixSummary.mean)
    //println("Movie factors variance: " + movieMatrixSummary.variance)
    //println("User factors mean: " + userMatrixSummary.mean)
    //println("User factors variance: " + userMatrixSummary.variance)
    val numClusters = 5
    val numIterations = 10
    val numRuns = 3
    val movieClusterModel = KMeans.train(movieVectors, numClusters, numIterations, numRuns)
    val movieClusterModelConverged = KMeans.train(
      movieVectors, numClusters, 100)
    val userClusterModel = KMeans.train(userVectors, numClusters, numIterations, numRuns)
    val movie1 = movieVectors.first
    val movieCluster = movieClusterModel.predict(movie1)
    //println(movieCluster)
    val predictions = movieClusterModel.predict(movieVectors)
    // println(predictions.take(10).mkString(","))
    val titilesWithFactors = titleAndGenres.join(movieFactors)
    val moviesAssigned = titilesWithFactors.map {
      case (id, ((title,
        genres), vector)) =>
        val pred = movieClusterModel.predict(vector)
        val clusterCentre = movieClusterModel.clusterCenters(pred)
        val dist = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
        (id, title, genres.mkString(" "), pred, dist)
    }
    val clusterAssginments = moviesAssigned.groupBy {
      case (id, title, genres, cluster, dist) => cluster
    }.collectAsMap
    for ((k, v) <- clusterAssginments.toSeq.sortBy(_._1)) {
      println(s"Cluster $k:")
      val m = v.toSeq.sortBy(_._5)
      println(m.take(20).map {
        case (_, title, genres, _, d) =>
          (title, genres, d)
      }.mkString("\n"))
      println("====\n")

    }
    /*Computing performance metrics on the MovieLens dataset*/
    val movieCost = movieClusterModel.computeCost(movieVectors)
    val userCost = userClusterModel.computeCost(userVectors)
    println("WCSS for movies: " + movieCost)
    println("WCSS for users: " + userCost)
    val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6, 0.4), 123)
    val trainMovies = trainTestSplitMovies(0)
    val testMovies = trainTestSplitMovies(1)
    val costMovies = Seq(2, 3, 4, 5, 10, 20).map { k =>
      (k, KMeans.
        train(trainMovies, numIterations, k, numRuns).computeCost(testMovies))
    }
    println("Movie clustering cross-validation:")
    costMovies.foreach { case (k, cost) => println(f"WCSS for K=$k id $cost%2.2f") }
    val trainTestSplitUsers = userVectors.randomSplit(Array(0.6, 0.4),
      123)
    val trainUsers = trainTestSplitUsers(0)
    val testUsers = trainTestSplitUsers(1)
    val costsUsers = Seq(2, 3, 4, 5, 10, 20).map { k =>
      (k,
        KMeans.train(trainUsers, numIterations, k,
          numRuns).computeCost(testUsers))
    }
    println("User clustering cross-validation:")
    costsUsers.foreach { case (k, cost) => println(f"WCSS for K=$k id$cost%2.2f") }
  }

}