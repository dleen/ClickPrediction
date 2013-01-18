import scala.io.Source

class DataSet(filename: String) {
	val url = getClass.getResource("/" + filename)

	val dataIterator = Source.fromURL(url).getLines

	// for(elem <- dataIterator)
	// 	parseLine(elem)

	val temp = dataIterator.next

	println(temp)
	parseLine(temp).map(println)

	def parseLine(line: String): Array[String] = {
		val splitOnPipe = line.split('|')
		
		splitOnPipe
	}

}

object HelloWorld extends App {
	val t = new DataSet("train.txt")
}