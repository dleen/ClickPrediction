    // // Update the weight vector 
    // def updateWeightsHashMap(y: Double, yHat: Double,
    //         line: DataLine,
    //         w: mutable.HashMap[Int, Double],
    //         eta: Double): mutable.HashMap[Int, Double] = {
    //     val coeff: Double = (y - yHat) * eta
    //     val (features, index) = line.featuresArray

    //     val fzip = index zip features
    //     val map = mutable.HashMap.empty[Int, Double]

    //     for (p <- fzip) map += p
    //     map
    // }
package main.scala.sgd

class LogisticRegressionHashed(data: DataSet, eta: Double, lambda: Double) {

    def hashToSign(s: String): Int = {
        if (s.hashCode % 2 == 0) -1
        else 1
    }

    def hashToRange(s: String, m: Int): Int = {
        val hashVal = s.hashCode % m
        if (hashVal < 0) m + hashVal
        else hashVal
    }

}