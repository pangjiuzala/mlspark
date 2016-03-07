

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import com.sun.org.apache.xalan.internal.xsltc.compiler.Whitespace
import scala.util.matching.Regex
import org.apache.spark.ml.feature.StopWords
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.{ SparseVector => SV }
import org.apache.spark.mllib.feature.IDF
import breeze.linalg.SparseVector
import breeze.linalg.norm
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.Word2Vec
object AdvanceText {
  def tokenize(line: String, regex: Regex, stopwords: String, rareTokens: String): Seq[String] = {
    line.split("""\W+""")
      .map(_.toLowerCase)
      .filter(token => regex.pattern.matcher(token).matches)
      .filterNot(token => stopwords.contains(token))
      .filterNot(token => rareTokens.contains(token))
      .filter(token => token.size >= 2)
      .toSeq
  }
  def main(args: Array[String]): Unit = {
    val path = "hdfs://master:9000/user/root/input/20news-bydate-train/*"
    val conf = new SparkConf().setAppName("text")
    val sc = new SparkContext(conf)
    val rdd = sc.wholeTextFiles(path)
    //    val text = rdd.map {
    //      case (file, text) => text
    //    }
    // println(text.count)
    val newgroups = rdd.map {
      case (file, text) =>
        file.split("/").takeRight(2).head
    }

    val countByGroup = newgroups.map(n => (n, 1)).reduceByKey(_ + _).collect.sortBy(-_._2).mkString("\n")
    //println(countByGroup)
    val text = rdd.map {
      case (file, text) => text
    }
    val whileSpaceSplit = text.flatMap(t => t.split(" ").
      map(_.toLowerCase))
    //println(whileSpaceSplit.distinct.count)
    // println(whileSpaceSplit.sample(true, 0.3, 42).take(100).mkString(","))
    val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))
    // println(nonWordSplit.distinct.count)
    //println(nonWordSplit.distinct.sample(true, 0.3, 42).take(100).mkString(","))
    val regex = """[^0-9]*""".r
    val filterNumbers = nonWordSplit.filter(token =>
      regex.pattern.matcher(token).matches)
    // println(filterNumbers.distinct.count)
    //println(filterNumbers.distinct.sample(true, 0.3, 42).take(100).mkString(","))
    val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + _)
    val oreringDesc = Ordering.by[(String, Int), Int](_._2)
    //println(tokenCounts.top(20)(oreringDesc).mkString("\n"))
    val stopwords = Set(
      "the", "a", "an", "of", "or", "in", "for", "by", "on", "but", "is", "not",
      "with", "as", "was", "if",
      "they", "are", "this", "and", "it", "have", "from", "at", "my",
      "be", "that", "to")
    val tokenCountsFilteredStopwords = tokenCounts.filter {
      case (k, v) =>
        !stopwords.contains(k)
    }
    //println(tokenCountsFilteredStopwords.top(20)(oreringDesc).mkString("\n"))
    val tokenCountsFiltereredSize = tokenCountsFilteredStopwords.filter {
      case (k, v) => k.size >= 2

    }
    //println(tokenCountsFiltereredSize.top(20)(oreringDesc).mkString("\n"))
    val rareTokens = tokenCounts.filter { case (k, v) => v < 2 }.map {
      case (k, v) => k
    }.collect.toSet
    val tokenCountsFilteredAll = tokenCountsFiltereredSize.filter {
      case (k, v) => !rareTokens.contains(k)
    }
    //println(tokenCountsFilteredAll.top(20)(oreringDesc).mkString("\n"))
    val tokens = text.map(doc => tokenize(doc, regex, stopwords.toString(), rareTokens.toString()))
    val dim = math.pow(2, 18).toInt
    val hashingTF = new HashingTF(dim)
    val tf = hashingTF.transform(tokens)
    tf.cache
    /*conculate tf*/
    val v = tf.first.asInstanceOf[SV]
    //println(v.size)
    //println(v.values.size)
    //println(v.values.take(10).toSeq)
    //println(v.indices.take(10).toSeq)
    /*conculate idf*/
    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf)
    val v2 = tfidf.first.asInstanceOf[SV]
    //println(v2.size)
    //println(v2.values.size)
    //println(v2.values.take(10).toSeq)
    //println(v2.indices.take(10).toSeq)
    val minMaxVals = tfidf.map { v =>
      val sv = v.asInstanceOf[SV]
      (sv.values.min, sv.values.max)
    }
    val globalMinMax = minMaxVals.reduce {
      case ((min1, max1), (min2, max2)) =>
        (math.min(min1, min2), math.max(max1, max2))
    }
    //println(globalMinMax)
    val common = sc.parallelize(Seq(Seq("you", "do", "we")))
    val tfCommon = hashingTF.transform(common)
    val tfidfCommon = idf.transform(tfCommon)
    val commonVector = tfidfCommon.first.asInstanceOf[SV]
    //println(commonVector.values.toSeq)
    val uncommon = sc.parallelize(Seq(Seq("telescope", "legislation",
      "investment")))
    val tfUncommon = hashingTF.transform(uncommon)
    val tfidfUncommon = idf.transform(tfUncommon)
    val uncommonVector = tfidfUncommon.first.asInstanceOf[SV]
    //println(uncommonVector.values.toSeq)
    val hockeyText = rdd.filter {
      case (file, text) => file.contains("honey")
    }
    val hockeyTF = hockeyText.mapValues(doc =>
      hashingTF.transform(tokenize(doc, regex, stopwords.toString(), rareTokens.toString())))
    val hockeyTFIdf = idf.transform(hockeyTF.map(_._2))
    val hockey1 = hockeyTFIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
    val breeze1 = new SparseVector(hockey1.indices, hockey1.values, hockey1.size)
    val hockey2 = hockeyTFIdf.sample(true, 0.1, 43).first.asInstanceOf[SV]
    val breeze2 = new SparseVector(hockey2.indices, hockey2.values, hockey2.size)
    val cosineSim = breeze1.dot(breeze2) / (norm(breeze1) * norm(breeze2))
    //println(cosineSim)
    val graphicsText = rdd.filter {
      case (file, text) =>
        file.contains("comp.graphics")
    }
    val graphicsTF = graphicsText.mapValues(doc =>
      hashingTF.transform(tokenize(doc, regex, stopwords.toString(), rareTokens.toString())))
    val graphicsTFIdf = idf.transform(graphicsTF.map(_._2))
    val graphics = graphicsTFIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
    val breezeGraphics = new SparseVector(graphics.indices, graphics.values, graphics.size)
    val cosineSim2 = breeze1.dot(breezeGraphics) / (norm(breeze1) * norm(breezeGraphics))
    println(cosineSim2)
    val baseballText = rdd.filter {
      case (file, text) =>
        file.contains("baseball")
    }
    val baseballTF = baseballText.mapValues(doc =>
      hashingTF.transform(tokenize(doc, regex, stopwords.toString(), rareTokens.toString())))
    val baseballTfIdf = idf.transform(baseballTF.map(_._2))
    val baseball = baseballTfIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
    val breezeBaseball = new SparseVector(baseball.indices,
      baseball.values, baseball.size)
    val consineSim3 = breeze1.dot(breezeBaseball) / (norm(breeze1) *
      norm(breezeBaseball))
    println(consineSim3)
    val newsgroupsMap = newgroups.distinct.collect().zipWithIndex.toMap
    val zipped = newgroups.zip(tfidf)
    val train = zipped.map { case (topic, vector) => LabeledPoint(newsgroupsMap(topic), vector) }
    train.cache
    val model = NaiveBayes.train(train, lambda = 0.1)
    val testPath = "hdfs://master:9000/user/root/input/20news-bydate-test/*"
    val testRDD = sc.wholeTextFiles(testPath)
    val testLabels = testRDD.map {
      case (file, text) =>
        val topic = file.split("/").takeRight(2).head
        newsgroupsMap(topic)
    }
    val testTf = testRDD.map {
      case (file, text) =>
        hashingTF.transform(tokenize(text, regex, stopwords.toString(), rareTokens.toString()))
    }
    val testTfIdf = idf.transform(testTf)
    val zippedTest = testLabels.zip(testTfIdf)
    val test = zippedTest.map {
      case (topic, vector) =>
        LabeledPoint(topic, vector)
    }
    val predictionAndLabel = test.map(p => (model.predict(p.features),
      p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    val metrics = new MulticlassMetrics(predictionAndLabel)
    println(accuracy)
    println(metrics.weightedFMeasure)
    val rawTokens = rdd.map { case (file, text) => text.split(" ") }
    //evaluate the performance on the test set as we did for the model trained with TF-IDF features:
    val rawTF = tokens.map(doc => hashingTF.transform(doc))
    val rawTrain = newgroups.zip(rawTF).map {
      case (topic, vector) =>
        LabeledPoint(newsgroupsMap(topic), vector)
    }
    val rawModel = NaiveBayes.train(rawTrain, lambda = 0.1)
    val rawTestTF = testRDD.map {
      case (file, text) =>
        hashingTF.transform(text.split(" "))
    }
    val rawZippedTest = testLabels.zip(rawTestTF)
    val rawTest = rawZippedTest.map {
      case (topic, vector) =>
        LabeledPoint(topic, vector)
    }
    val rawPredictionAndLabel = rawTest.map(p =>
      (rawModel.predict(p.features), p.label))
    val rawAccuracy = 1.0 * rawPredictionAndLabel.filter(x => x._1 ==
      x._2).count() / rawTest.count()
    println(rawAccuracy)
    val rawMetrics = new MulticlassMetrics(rawPredictionAndLabel)
    println(rawMetrics.weightedFMeasure)
    val word2vec = new Word2Vec()
    word2vec.setSeed(42)
    val word2vecModel = word2vec.fit(tokens)
    word2vecModel.findSynonyms("hockey", 20).foreach(println)
    word2vecModel.findSynonyms("legislation", 20).foreach(println)
  }

}