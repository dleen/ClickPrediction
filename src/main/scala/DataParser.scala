package main.scala.sgd

import scala.io.Source

// Object to store the results of parsing one line
case class DataLine(clicked: Int,
    depth: Int,
    position: Int,
    userid: Int,
    gender: Int,
    age: Int,
    tokens: Set[Int]) {

    // Creates the feature array x which includes the 1 
    // corresponding to w0 i.e. x = (1, x)
    def featuresArray(): (Array[Double], Array[Int]) = {
        val features = Array(1.0, // Corresponds to w0
            this.depth.toDouble,
            this.position.toDouble,
            this.gender.toDouble,
            this.age.toDouble) ++ Array.fill(this.tokens.size)(1.0)
        val index = Array(0, 1, 2, 3, 4) ++
            this.tokens.toArray.map(x => x + 5).sorted
        (features, index)
    }
}


// Object representing one dataset
case class DataSet(datatype: String) {
    val filename: String = datatype match {
        case "training" => "train.txt"
        case "test" => "test.txt"
    }

    // Open resource, either training or test data
    // and create iterator which iterates over each
    // line and returns a DataLine 
    val url = getClass.getResource("/" + filename)
    val dataFile = Source.fromURL(url)
    val dataIterator = dataFile.getLines.map(parseLine)

    val tokensLength: Int = datatype match {
        case "training" => 141063
        case "test" => 109459
    }
    val maxTokenValue: Int = datatype match {
        case "training" => 1070659
        case "test" => 1070634
    }
    val numOfLines: Int = datatype match {
        case "training" => 2335860
        case "test" => 1016553
    }
    val offset: Int = 5

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

case object TestLabels {
    val url = getClass.getResource("/" + "test_label.txt")
    val dataFile = Source.fromURL(url)
    val label = dataFile.getLines.map(_.toDouble)

    def closeData = {
        dataFile.close
    }
}