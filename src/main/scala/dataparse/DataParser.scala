package main.scala.dataparse

import scala.io.Source

case class DataLine(clicked: Int,
    depth: Int,
    position: Int,
    userid: Int,
    gender: Int,
    age: Int,
    tokens: Set[Int])

case class DataSet(datatype: String) {
    val filename: String = datatype match {
        case "training" => "train.txt"
        case "test" => "test.txt"
        case "test_labels" => "test_labels.txt"
    }

    val url = getClass.getResource("/" + filename)
    val dataFile = Source.fromURL(url)
    val dataIterator = dataFile.getLines.map(parseLine)

    def parseLine(line: String): DataLine = {
        val splitOnPipe = line.split('|')
        
        val nonToken = splitOnPipe.init.map(_.toInt)
        val tokens = parseTokens(splitOnPipe.last)

        datatype match {
            case "training" => DataLine(nonToken(0),
                nonToken(1), nonToken(2), nonToken(3),
                nonToken(4), nonToken(5), tokens)
            case "test" => DataLine(-1,
                nonToken(0), nonToken(1), nonToken(2),
                nonToken(3), nonToken(4), tokens)
        }
    }

    def parseTokens(tokenString: String): Set[Int] = {
        val splitOnComma = tokenString.split(',').map(_.toInt)

        splitOnComma.toSet
    }

    def resetDataIterator = {
        if (this.dataIterator.hasNext) this
        else new DataSet(datatype)
    }

    def closeData = {
        dataFile.close
    }
}