package main.scala.dataparse

import scala.io.Source

// Object to store the results of parsing one line
case class DataLine(clicked: Int,
    depth: Int,
    position: Int,
    userid: Int,
    gender: Int,
    age: Int,
    tokens: Set[Int])

// Object representing one dataset
case class DataSet(datatype: String) {
    val filename: String = datatype match {
        case "training" => "train.txt"
        case "test" => "test.txt"
        case "test_labels" => "test_labels.txt"
    }

    // Open resource, either training or test data
    // and create iterator which iterates over each
    // line and returns a DataLine 
    val url = getClass.getResource("/" + filename)
    val dataFile = Source.fromURL(url)
    val dataIterator = dataFile.getLines.map(parseLine)

    // Parses a string into a DataLine object
    def parseLine(line: String): DataLine = {
        val splitOnPipe = line.split('|')
        // The first n - 1 elements are not tokens
        val nonToken = splitOnPipe.init.map(_.toInt)
        // The last element is a token
        // We parse the token into a set of ints
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

    def resetDataIterator: DataSet = {
        if (this.dataIterator.hasNext) this
        else new DataSet(datatype)
    }

    def closeData = {
        dataFile.close
    }
}