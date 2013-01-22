import main.scala.dataparse._

case class BasicAnalysis(datatype: String,
	meanCTR: Double,
    uniqueTokens: collection.mutable.Set[Int],
    uniqueUsers: collection.mutable.Set[Int],
    n: Int) {

    def intersect(that: BasicAnalysis) = {
        val commonTokens = this.uniqueTokens.intersect(that.uniqueTokens) 
        val commonUsers = this.uniqueUsers.intersect(that.uniqueUsers)

        (commonTokens, commonUsers)
    }

    override def toString = "Data set: " + datatype + ", CTR: " + meanCTR.toString.take(5) +
    	", # Users: " + uniqueUsers.size + ", # Tokens: " + 
    	uniqueTokens.size + ", # Lines: " + n
}

object WarmUp extends App {
    val training = DataSet("training")
    val test = DataSet("test")

    val summaryTraining = warmUpCalculation(training)
    val summaryTest = warmUpCalculation(test)

    val (commonTokens, commonUsers) = summaryTraining.intersect(summaryTest)

    println(summaryTraining)
    println(summaryTest)
    println("# Tokens in both data sets: " + commonTokens.size)
    println("# Users in both data sets: " + commonUsers.size) 

    def warmUpCalculation(data: DataSet): BasicAnalysis = {
        var mean = 0.0
        var n = 1

        var users = collection.mutable.Set.empty[Int]
        var tokens = collection.mutable.Set.empty[Int]

        for (x <- data.dataIterator) {
            mean = ((n - 1) * mean + x.clicked) / n
            n = n + 1

        	if (x.userid != 0) {
	            users += x.userid
	        }
	        tokens ++= x.tokens
        }

        data.closeData

        BasicAnalysis(data.datatype, mean, tokens, users, n)
    }
}