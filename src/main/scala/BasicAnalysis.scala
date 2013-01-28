import main.scala.dataparse._
// Stores the results of the warm up basic analysis
case class BasicAnalysis(datatype: String,
	meanCTR: Double,
    uniqueTokens: collection.mutable.Set[Int],
    uniqueUsers: collection.mutable.Set[Int],
    n: Int) {
    // Finds the intersections, elements common to both,
    // between the users and tokens sets of different BasicAnalysis objects
    def intersect(that: BasicAnalysis) = {
        val commonTokens = this.uniqueTokens.intersect(that.uniqueTokens) 
        val commonUsers = this.uniqueUsers.intersect(that.uniqueUsers)
        (commonTokens, commonUsers)
    }

    override def toString = "Data set: " + datatype + ", CTR: " + 
        meanCTR.toString.take(10) + ", # Users: " + uniqueUsers.size + 
        ", # Tokens: " + uniqueTokens.size + ", # Lines: " + n
}

object WarmUp extends App {
    // Creates iterators for training and test sets
    val training = DataSet("training")
    val test = DataSet("test")
    // Calculate summaries for both data sets
    val summaryTraining = warmUpCalculation(training)
    val summaryTest = warmUpCalculation(test)
    // Calculate common elements in the training and test
    // users and tokens sets
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

        // Loop over the lines in the data
        for (x <- data.dataIterator) {
            // Single pass mean of CTR
            mean += x.clicked
            n = n + 1
        	if (x.userid != 0) {
                // Add unique userids to the set
	            users += x.userid
	        }
            // Add sets of unique tokens to the token set
	        tokens ++= x.tokens
        }
        mean = mean / (n - 1)
        // Close data file
        data.closeData
        // Return the basic analysis
        BasicAnalysis(data.datatype, mean, tokens, users, n)
    }
}