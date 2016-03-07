
package sparkstream
import scala.util.Random
import java.net.ServerSocket
import java.io.PrintWriter

object StreamingProducer {
  def main(args: Array[String]) {

    val random = new Random()
    //Maximum number of events per second
    val MaxEvents = 6
    //Read the list of possible names
    val namesResource = this.getClass.getResourceAsStream("/names.csv")
    val names = scala.io.Source.fromInputStream(namesResource)
      .getLines()
      .toList
      .head
      .split(",")
      .toSeq

    //Generate a sequence of possible products
    val products = Seq(
      "iphone Cover" -> 9.99,
      "Headphones" -> 5.49,
      "Samsung Galaxy Cover" -> 8.95,
      "iPad Cover" -> 7.49)

    def generateProductEvents(n: Int) = {
      (1 to n).map {
        i =>
          val (product, price) =
            products(random.nextInt(products.size))
          val user = random.shuffle(names).head
          (user, product, price)
      }
    }
    val listener = new ServerSocket(9999)
    println("Listening on port:9999")
    while (true) {
      val socket = listener.accept()
      new Thread() {
        override def run = {
          println("Got client connected from:" +
            socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream(), true)
          while (true) {
            Thread.sleep(1000)
            val num = random.nextInt(MaxEvents)
            val productEvents = generateProductEvents(num)
            productEvents.foreach {
              event =>
                out.write(event.productIterator.mkString(","))
                out.write("\n")
            }
            out.flush()
            println(s"Created $num events..")
          }
          socket.close()
        }
      }.start()
    }
  }
}