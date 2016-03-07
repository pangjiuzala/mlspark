

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object RegressionTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Regression")
    val sc = new SparkContext(conf)
    val raw_data = sc.textFile("hdfs://master:9000/user/root/mllib/hour.csv")
    val num_data = raw_data.count()
    val records = raw_data.map(line => line.split(","))
    val first = records.first()
    println(first.toString())
    println(num_data)

  }
}