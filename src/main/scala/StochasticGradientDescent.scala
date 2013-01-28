package main.scala.sgd

import main.scala.dataparse._

import scala.annotation.tailrec
import scala.math.{exp, sqrt, pow}

import breeze.linalg._

object SGD extends App {

    val training = DataSet("training")
    val eta = 0.001

    val LR = new LogisticRegression(training, eta)

    println("The L2 norm: " + l2Norm(LR.weightsFinal))
    println("Max of weights: " + LR.weightsFinal.max)
    println("The avg loss list: " + LR.avgLossListFinal.length)
    println("The avg loss first: " + LR.avgLossListFinal.head)
    println("The avg loss 10,000: " + LR.avgLossListFinal(10000))
    println("The avg loss last: " + LR.avgLossListFinal.last)

    training.closeData

    def l2Norm(w: SparseVector[Double]): Double = w.norm(2)
}

class LogisticRegression(data: DataSet, eta: Double) {
    val offset = data.offset
    val maxTokenValue = data.maxTokenValue

    val (weightsFinal, avgLossListFinal) = calculateWeights()

    def calculateWeights(): (SparseVector[Double], List[Double]) = {
    	println("Calculating the weights: ")
        val avgLossList: List[Double] = List()
        val avgLoss: Double = 0.0
        val w = SparseVector.zeros[Double](maxTokenValue + offset + 1)

        @tailrec def recursiveSGD(iter: Iterator[DataLine],
            weights: SparseVector[Double],
            eta: Double,
            avgLoss: Double,
            avgLossList: List[Double],
            n: Int): (SparseVector[Double], List[Double]) = {
            if (!iter.hasNext) (weights, avgLossList)
            else {
                // A line from the data set
                val line = iter.next
                // Create the feature vector
                val x: SparseVector[Double] = featureVector(line)
                // The actual label, clicked or not clicked
                val y: Int = line.clicked
                // The predicted label P(Y=1|X)
                val yHat: Double = predictLabel(x, weights)
                // Calculate the new weights using SGD
                val newWeights: SparseVector[Double] = 
                	updateWeights(y, yHat, x, weights, eta)
                // Calculate the average loss, the square diff between
                // actual and predicted labels
                val avgLossNew: Double = avgLossFunc(avgLoss, y, yHat, n)
                // A list of every 100th average loss value
                val avgLossListNew: List[Double] = 
                	updateAvgLossList(avgLossNew, avgLossList, n)
                // Move onto the next line to update the weights
                recursiveSGD(iter, newWeights, eta, avgLossNew, avgLossListNew, n + 1)
            }
        }
        recursiveSGD(data.dataIterator, w, eta, avgLoss, avgLossList, 1)
    }

    // Return the feature vector X from the data in a line
    def featureVector(line: DataLine): SparseVector[Double] = {
        val (features, index) = line.featuresArray
        new SparseVector(index, features, index.size, maxTokenValue + offset + 1) 
    }

    // Return the average loss
    def avgLossFunc(avgLossPrev: Double, y: Double, 
            yHat: Double, n:Int): Double = {
        ((n - 1) * avgLossPrev + pow(y - yHat, 2)) / n
    }

    // Update the loss list with every 100th value
    def updateAvgLossList(avgLoss: Double, avgLossList: List[Double],
            n: Int): List[Double] = {
        if (n % 100 == 0) {
            System.out.print("\rPercent done: " + (100 * n / data.numOfLines.toDouble).toString.take(4))
            System.out.flush()
            // Thread.sleep(100)
        	// print((100 * n / data.numOfLines.toDouble).toString.take(4) + ", ")
        	avgLoss :: avgLossList
        }
        else avgLossList
    }

    // Update the weight vector 
    def updateWeights(y: Double, yHat: Double,
            x: SparseVector[Double],
            w: SparseVector[Double],
            eta: Double): SparseVector[Double] = {
        val coeff: Double = (y - yHat) * eta
        // val xNew: SparseVector[Double] = x * coeff
        x *= coeff
        w += x
        w
    }

    // Make a prediction for the label
    def predictLabel(x: SparseVector[Double],
            w: SparseVector[Double]): Double = {
        val ewx = exp(w.dot(x))
        ewx / (1 + ewx)
    }
}
