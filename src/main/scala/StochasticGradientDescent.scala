package main.scala.sgd

import main.scala.dataparse._

import scala.collection._
import scala.annotation.tailrec
import scala.math.{exp, sqrt, pow}

import breeze.linalg._

object SGD extends App {

    val training = DataSet("training")
    val eta = 0.05
    val lambda = 0.0005

    val LR = new LogisticRegression(training, eta, lambda)
    val (weightsFinal, avgLossListFinal) = LR.calculateWeights()

    println("The L2 norm: "         + l2Norm(weightsFinal))
    println("Max of weights: "      + weightsFinal.max)
    println("The avg loss list: "   + avgLossListFinal.length)
    println("The avg loss first: "  + avgLossListFinal.head)
    println("The avg loss 10,000: " + avgLossListFinal(10000))
    println("The avg loss last: "   + avgLossListFinal.last)

    training.closeData

    def l2Norm(w: SparseVector[Double]): Double = w.norm(2)
}

class LogisticRegression(data: DataSet, eta: Double, lambda: Double) {
    val offset = data.offset
    val maxTokenValue = data.maxTokenValue

    def predictCTR(test: Iterator[DataLine],
        labels: Iterator[TestLabels],
        weights: SparseVector[Double],
        baseLine: Double,
        n: Int,
        rmse: Double,
        rmseBaseLine: Double): (Double, Double) = {
        if (!test.hasNext && !labels.hasNext) 
        }
        predictCTR(test, labels, weights, baseLine)
    }


    def calculateWeights(): (SparseVector[Double], List[Double]) = {
    	println("Calculating the weights: ")
        val avgLossList: List[Double] = List()
        val avgLoss: Double = 0.0
        val w = SparseVector.zeros[Double](maxTokenValue + offset + 1)
        val aTimes = SparseVector.zeros[Int](maxTokenValue + offset + 1)

        @tailrec def recursiveSGD(iter: Iterator[DataLine],
            weights: SparseVector[Double],
            accessTimes: SparseVector[Int],
            avgLoss: Double,
            avgLossList: List[Double],
            n: Int): (SparseVector[Double], List[Double]) = {
            if (!iter.hasNext) (weights, avgLossList)
            else {
                // A line from the data set
                val line = iter.next
                // Create the feature vector
                val x: SparseVector[Double] = featureVector(line)
                // Update access times
                val newAccessTimes: SparseVector[Int] =
                    calcAccessTimes(line, accessTimes, n)
                // Perform the delayed regularization
                val (delayRegWeights, delayAccessTimes) = 
                    performDelayedReg(newAccessTimes, x, weights, n)
                // The actual label, clicked or not clicked
                val y: Int = line.clicked
                // The predicted label P(Y=1|X)
                val yHat: Double = predictLabel(x, delayRegWeights)
                // Calculate the average loss, the square diff between
                // actual and predicted labels
                val avgLossNew: Double = avgLossFunc(avgLoss, y, yHat, n)
                // A list of every 100th average loss value
                val avgLossListNew: List[Double] = 
                    updateAvgLossList(avgLossNew, avgLossList, n)
                // Calculate the new weights using SGD
                val newWeights: SparseVector[Double] = 
                	updateWeights(y, yHat, x, delayRegWeights, eta)
                // Perform the standard regularization
                val (stdRegWeights, stdAccessTimes) = 
                    performStandardReg(delayAccessTimes, x, newWeights, n)
                // Move onto the next line to update the weights
                recursiveSGD(iter, stdRegWeights, stdAccessTimes, avgLossNew, avgLossListNew, n + 1)
            }
        }
        recursiveSGD(data.dataIterator, w, aTimes, avgLoss, avgLossList, 1)
    }

    // Return the feature vector X from the data in a line
    def featureVector(line: DataLine): SparseVector[Double] = {
        val (features, index) = line.featuresArray
        new SparseVector(index, features, index.size, maxTokenValue + offset + 1) 
    }

    def calcAccessTimes(line: DataLine,
            accessTimes: SparseVector[Int],
            n: Int): SparseVector[Int] = {
        val (_, index) = line.featuresArray
        for (i <- index)
            if (accessTimes(i) == 0) accessTimes(i) = n // First appearance

        accessTimes
    }

    def performDelayedReg(accessTimes: SparseVector[Int],
            x: SparseVector[Double],
            w: SparseVector[Double],
            n: Int): (SparseVector[Double], SparseVector[Int]) = {
        val coeff = 1 - (eta * lambda)
        for (i <- x.keysIterator) {
            val t1 = accessTimes(i)
            if (t1 != 0 && n - t1 > 1) {
                w.update(i, w(i) * pow(coeff, n - t1 - 1))
                accessTimes.update(i, n)
            }
        }
        (w, accessTimes)
    }

    def performStandardReg(accessTimes: SparseVector[Int],
            x: SparseVector[Double],
            w: SparseVector[Double],
            n: Int): (SparseVector[Double], SparseVector[Int]) = {
        val coeff = 1 - (eta * lambda)
        for (i <- x.keysIterator) {
            val t1 = accessTimes(i)
            if (t1 != 0 && t1 == n - 1) {
                w.update(i, w(i) * coeff)
                accessTimes.update(i, n)
            }
        }
        (w, accessTimes)
    }

    // Update the weight vector 
    def updateWeights(y: Double, yHat: Double,
            x: SparseVector[Double],
            w: SparseVector[Double],
            eta: Double): SparseVector[Double] = {
        val coeff: Double = (y - yHat) * eta
        x *= coeff
        w += x
        w
    }

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

    // Make a prediction for the label
    def predictLabel(x: SparseVector[Double],
            w: SparseVector[Double]): Double = {
        val ewx = exp(w.dot(x))
        ewx / (1 + ewx)
    }

    // Return the average loss
    def avgLossFunc(avgLossPrev: Double, y: Double, 
            pyx: Double, n:Int): Double = {
    	val yHat = {
    		if (pyx >= 0.5) 1
    		else 0
    	}
        ((n - 1) * avgLossPrev + pow(y - yHat, 2)) / n
    }

    // Update the loss list with every 100th value
    def updateAvgLossList(avgLoss: Double, avgLossList: List[Double],
            n: Int): List[Double] = {
        if (n % 100 == 0) {
            System.out.print("\rPercent done: " + (100 * n / data.numOfLines.toDouble).toString.take(4))
            System.out.flush()
        	avgLoss :: avgLossList
        }
        else avgLossList
    }

}
