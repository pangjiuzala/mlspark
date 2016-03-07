

import org.apache.spark.SparkConf
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
object KafkaWordCount {
  
  def main(args: Array[String]): Unit = {
    if(args.length<4){
      System.exit(1)
    }
    val Array(zkQuorm,group,topics,numThreads)=args
    val sparkConf=new SparkConf().setAppName("KafkaWordcount")
    val ssc=new StreamingContext(sparkConf,Seconds(2))
    ssc.checkpoint("checkpoint")
    val topicMap=topics.split(",").map((_,numThreads.toInt)).toMap
    val lines=KafkaUtils.createStream(ssc,zkQuorm,group,topicMap).map(_._2)
    val words=lines.flatMap(_.split(" "))
    val wordCounts=words.map(x=>(x,1L))
   .reduceByKeyAndWindow(_+_,_-_,Seconds(10),Seconds(2),2)
    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }
}