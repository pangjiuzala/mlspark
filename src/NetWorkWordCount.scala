

import org.apache.spark.SparkConf
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds

object NetWorkWordCount {
  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: NetworkWordCount <hostname> <port>")
      System.exit(1)
    }
  }
  val sparkConf = new SparkConf().setAppName("NetworkWordCount")
  val ssc = new StreamingContext(sparkConf, Seconds(1))
  val lines=ssc.socketTextStream("master", 9999)
  val words=lines.flatMap(_.split(""))
  val wordCounts=words.map (x => (x,1)).reduceByKey(_+_)
  wordCounts.print()
  ssc.start()
  ssc.awaitTermination()
}