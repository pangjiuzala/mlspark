

import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object WordCounts {
  def main(args: Array[String]) {
    val logFile = "hdfs://master:9000/user/root/input/file2.txt" // Should be some file on your system
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    // val tabletesta = sqlContext.sql("SELECT count(a.ID) num from tablea a join tablea b on to_int(a.ID) < to_int(b.ID) ")
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
    val numAss = logData.filter(line => line.contains("a")).count();

  }
}