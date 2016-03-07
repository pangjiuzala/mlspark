

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j.Level
object Kmeans {
  def main(args: Array[String]): Unit = {

    // 屏蔽不必要的日志显示在终端上

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    // 设置运行环境

    val conf = new SparkConf().setAppName("Kmeans").setMaster("local[4]")

    val sc = new SparkContext(conf)

    // 装载数据集

    val data = sc.textFile("hdfs://master:9000/user/root/mllib/kmeans_data.txt", 1)

    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))

    // 将数据集聚类，2个类，20次迭代，进行模型训练形成数据模型

    val numClusters = 2

    val numIterations = 20

    val model = KMeans.train(parsedData, numClusters, numIterations)

    // 打印数据模型的中心点

    println("Cluster centers:")

    for (c <- model.clusterCenters) {

      println("  " + c.toString)

    }

    // 使用误差平方之和来评估数据模型

    val cost = model.computeCost(parsedData)

    println("Within Set Sum of Squared Errors = " + cost)

    // 使用模型测试单点数据

    println("Vectors 0.2 0.2 0.2 is belongs to clusters:" + model.predict(Vectors.dense("0.2 0.2 0.2".split(' ').map(_.toDouble))))

    println("Vectors 0.25 0.25 0.25 is belongs to clusters:" + model.predict(Vectors.dense("0.25 0.25 0.25".split(' ').map(_.toDouble))))

    println("Vectors 8 8 8 is belongs to clusters:" + model.predict(Vectors.dense("8 8 8".split(' ').map(_.toDouble))))

    // 交叉评估1，只返回结果

    val testdata = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))

    val result1 = model.predict(testdata)

    result1.saveAsTextFile("hdfs://master:9000/user/root/mllib/result_kmeans1")

    // 交叉评估2，返回数据集和结果

    val result2 = data.map {

      line =>

        val linevectore = Vectors.dense(line.split(' ').map(_.toDouble))

        val prediction = model.predict(linevectore)

        line + " " + prediction

    }.saveAsTextFile("hdfs://master:9000/user/root/mllib/result_kmeans2")

    sc.stop()
  }
}