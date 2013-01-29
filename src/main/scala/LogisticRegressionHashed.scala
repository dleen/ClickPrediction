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

import scala.collection._

object HashedLR extends App {
    val training = DataSet("training")

}


class LogisticRegressionHashed(data: DataSet, eta: Double, lambda: Double) {

    val weights = mutable.Map[String, Int]()

    def hashedFeatureList(fl: List[String])

    // Xi
    def hashToSign(s: String): Int = {
        if (s.hashCode % 2 == 0) -1
        else 1
    }

    // h
    def hashToRange(s: String, m: Int): Int = {
        val hashVal = s.hashCode % m
        if (hashVal < 0) m + hashVal
        else hashVal
    }

    def featureList(l: DataLine): List[String] = l.featuresString

}