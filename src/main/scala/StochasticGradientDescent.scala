package main.scala.sgd

import main.scala.dataparse._

import scala.annotation.tailrec
import scala.math._

import breeze.linalg._

object SGD extends App {

    val training = DataSet("training")
    val eta = 0.05

    val LR = new LogisticRegression(training, eta)
    val (wFin, avgLossListFin) = LR.calculateWeights()

    println("The L2 norm: " + l2Norm(wFin))
    println("The avg loss list: " + avgLossListFin.length)
    println("The avg loss first: " + avgLossListFin.head)
    println("The avg loss last: " + avgLossListFin.last)

    training.closeData

    def l2Norm(w: SparseVector[Double]): Double = {
        sqrt(w.dot(w))
    }    
}

class LogisticRegression(data: DataSet, eta: Double) {
    val offset = data.offset
    val maxTokenValue = data.maxTokenValue
    val z = data.dataIterator

    def calculateWeights(): (SparseVector[Double], List[Double]) = {
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
                val x = featureVector(line)
                // The actual label, clicked or not clicked
                val y = line.clicked
                // The predicted label P(Y=1|X)
                val yHat = predictLabel(x, weights)
                // Calculate the new weights using SGD
                val newWeights = updateWeights(y, yHat, x, weights, eta)
                // Calculate the average loss, the square diff between
                // actual and predicted labels
                val avgLossNew = avgLossFunc(avgLoss, y, yHat, n)
                // A list of every 100th average loss value
                val avgLossListNew = updateAvgLossList(avgLossNew, avgLossList, n)

                // Move onto the next line to update the weights
                recursiveSGD(iter, newWeights, eta, avgLossNew, avgLossListNew, n + 1)
            }
        }
        recursiveSGD(z, w, eta, avgLoss, avgLossList, 1)
    }

    // Return the feature vector X from the data in a line
    def featureVector(line: DataLine): SparseVector[Double] = {
        val (features, index) = line.featuresArray
        new SparseVector(index, features, maxTokenValue + offset + 1) 
    }

    // Return the average loss
    def avgLossFunc(avgLossPrev: Double, 
            y: Double, 
            yHat: Double, 
            n:Int): Double = {
        ((n - 1) * avgLossPrev + pow(y - yHat, 2)) / n
    }

    // Update the loss list with every 100th value
    def updateAvgLossList(avgLoss: Double,
            avgLossList: List[Double],
            n: Int): List[Double] = {
        if (n % 100 == 0) avgLoss :: avgLossList
        else avgLossList
    }

    // Update the weight vector 
    def updateWeights(y: Double, 
            yHat: Double,
            x: SparseVector[Double],
            w: SparseVector[Double],
            eta: Double): SparseVector[Double] = {
        val coeff = (y - yHat) * eta
        val xNew = x * coeff
        println(xNew.data.size, w.data.size)
        w :+ xNew
    }

    // Make a prediction for the label
    def predictLabel(x: SparseVector[Double],
            w: SparseVector[Double]): Double = {
        val logisticRegressExp = exp(x.dot(w))
        logisticRegressExp / (1 + logisticRegressExp)
    }
}

object smalltest extends App {
    val w = new SparseVector(Array(0,1,2),Array(0,1,2),3)
    val z = new SparseVector(Array(0,1,2),Array(3,4,5),3)

    println(w + z * 2 * 1)
    // w.map(println)

}