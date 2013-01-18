import scala.io.Source

class DataSet(filename: String) {
	val url = getClass.getResource("/" + filename)

	println(url)

	val dataIterator = Source.fromURL(url).getLines

	// for(elem <- dataIterator )

	println(dataIterator.next)

}

object HelloWorld extends App {
	val t = new DataSet("test.txt")
}