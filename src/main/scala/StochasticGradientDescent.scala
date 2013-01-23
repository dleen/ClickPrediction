import main.scala.dataparse._

import breeze.linalg._

object SGD extends App {
	println("Hello World")

    val training = DataSet("training")

    val z = training.dataIterator.next

    println(z)

}