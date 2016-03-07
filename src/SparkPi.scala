

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.math.random
object SparkPi {
  
  def main(args: Array[String]): Unit = {
    val conf=new SparkConf().setAppName("Spark Pi")
    val spark=new SparkContext(conf)
    val slices=if(args.length>0) args(0).toInt else 2
    val n=10000*slices
    val counts=spark.parallelize(1 to n,slices).map{i=>
      val x=random*2-1
      val y=random*2-1
      if(x*x+y*y<1) 1 else 0
    }.reduce(_+_)
    println("Pi is roughly "+4.0*counts/n)
    spark.stop()
  }
  
}