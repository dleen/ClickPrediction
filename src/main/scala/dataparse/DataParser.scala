import scala.io.Source

case class DataLine(clicked: Int,
    depth: Int,
    position: Int,
    userid: Int,
    gender: Int,
    age: Int,
    tokens: Set[Int])

case class BasicAnalysis(meanCTR: Double,
    uniqueTokens: collection.mutable.Set[Int],
    uniqueUsers: collection.mutable.Set[Int],
    n: Int) {

    def intersect(that: BasicAnalysis) = {
        val commonTokens = this.uniqueTokens.intersect(that.uniqueTokens) 
        val commonUsers = this.uniqueUsers.intersect(that.uniqueUsers)

        (commonTokens, commonUsers)
    }
}

case class DataSet(datatype: String) {
    val filename: String = datatype match {
        case "training" => "train.txt"
        case "test" => "test.txt"
        case "test_labels" => "test_labels.txt"
    }

    println(filename)

    val url = getClass.getResource("/" + filename)

    val dataIterator = Source.fromURL(url).getLines.map(parseLine)

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
}

object HelloWorld extends App {
    val t = new DataSet("training")
    val u = new DataSet("test")

    def calcMean(data: DataSet): BasicAnalysis = {
        var mean = 0.0
        var n = 1

        var users = collection.mutable.Set.empty[Int]
        var tokens = collection.mutable.Set.empty[Int]

        for (x <- data.dataIterator) {
            mean = ((n - 1) * mean + x.clicked) / n
            n = n + 1

            users += x.userid
            tokens ++= x.tokens
        }

        BasicAnalysis(mean, tokens, users, n)
    }

    val bat = calcMean(t)
    val bau = calcMean(u)

    println(bat.uniqueUsers.size)
    println(bau.uniqueUsers.size)

    println(bat.uniqueTokens.size)
    println(bau.uniqueTokens.size)

    val (ct, cu) = bat.intersect(bau)

    println(ct.size)
    println(cu.size) 
}