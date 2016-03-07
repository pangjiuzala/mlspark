
package sparkstream

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import java.text.SimpleDateFormat
import org.apache.hadoop.hive.metastore.api.Date
object StreamingAnalyticsApp {
  def main(args: Array[String]) {
    val ssc = new StreamingContext("master",
      "First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("master", 9999)
    // here we simply print out the first few elements of each
    // batch
    stream.print()

    //create stream of events from raw text elements
    val events = stream.map {
      record =>
        val event = record.split(",")
        (event(0), event(1), event(2))
    }
    events.foreachRDD { (rdd, time) =>
      val numPurchases = rdd.count()
      val uniqueUsers = rdd.map {
        case (user, _, _) => user
      }.distinct().count()
      val totalRevenue = rdd.map { case (_, _, price) => price.toDouble }.sum()
      val productsByPopularity = rdd.
        map { case (user, product, price) => (product, 1) }
        .reduceByKey(_ + _)
        .collect()
        .sortBy(-_._2)
      val mostPopular = productsByPopularity(0)
      val formatter = new SimpleDateFormat
      val dateStr = formatter.format(new Date(time.milliseconds))
      println(s"== Batch start time: $dateStr ==")
      println("Total purchases: " + numPurchases)
      println("Unique users: " + uniqueUsers)
      println("Total revenue: " + totalRevenue)
      println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))
    }
    ssc.start()
    ssc.awaitTermination()
  }
}