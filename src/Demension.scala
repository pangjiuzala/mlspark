

import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.SparkContext
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.csvwrite
import breeze.linalg.DenseMatrix
object Demension {
  def loadImageFromFile(path: String): BufferedImage = {
    ImageIO.read(new File(path))

  }
  def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics()
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    bwImage
  }
  def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0, 0, width, height, pixels)
  }
  def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }
  def approxEqual(array1: Array[Double], array2: Array[Double],
                  tolerance: Double = 1e-6): Boolean = {
    // note we ignore sign of the principal component / singularvector elements
    val bools = array1.zip(array2).map {
      case (v1, v2) => if (math.abs(math.abs(v1) - math.abs(v2)) > 1e-6) false else true
    }
    bools.fold(true)(_ & _)
  }
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DimensionReduce")
    val sc = new SparkContext(conf)
    val rdd = sc.wholeTextFiles("hdfs://master:9000/user/root/input/lfw-a/lfw/*")
    val first = rdd.first
    //println(first)
    val files = rdd.map {
      case (fileName, content) => fileName.replace("file:", "")
    }
    //println(files.first)
    //println(files.count)
    //val aePath = "hdfs://master:9000/user/root/input/lfw-a/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    //val aeImage = loadImageFromFile(aePath)
    //val grayImage = processImage(aeImage, 100, 100)
    //ImageIO.write(grayImage, "jpg", new File("hdfs://master:9000/user/root/input/aeGray.jpg"))
    val pixels = files.map(f => extractPixels(f, 50, 50))
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))
    val vectors = pixels.map(p => Vectors.dense(p))
    vectors.setName("image-vectors")
    vectors.cache
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    val scaledVectors = vectors.map(v => scaler.transform(v))
    //PCA
    val matrix = new RowMatrix(scaledVectors)
    val K = 10
    val pc = matrix.computePrincipalComponents(K)
    val rows = pc.numRows
    val cols = pc.numCols
    println(rows, cols)
    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
    //    csvwrite(new File("/tmp/pc.csv"), pcBreeze)
    val svd = matrix.computeSVD(10, computeU = true)
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
    println(s"S dimension: (${svd.s.size}, )")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")
    println(approxEqual(Array(1.0, 2.0, 3, 0), Array(1.0, 2.0, 3.0)))
    val breezeS = breeze.linalg.DenseVector(svd.s.toArray)
    val projectedSVD = svd.U.rows.map { v =>
      val breezeV = breeze.linalg.DenseVector(v.toArray)
      val multV = breezeV :* breezeS
      Vectors.dense(multV.data)
    }
    //projected.rows.zip(projectedSVD).map { case (v1, v2) =>approxEqual(v1.toArray, v2.toArray) }.filter(b => true).count
    val sValues = (1 to 5).map { i => matrix.computeSVD(i, computeU = false).s }
    sValues.foreach(println)
    val svd300 = matrix.computeSVD(300, computeU = false)
    val sMatrix = new DenseMatrix(1, 300, svd300.s.toArray)
    csvwrite(new File("/tmp/s.csv"), sMatrix)
  }
}