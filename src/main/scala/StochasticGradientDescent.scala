package main.scala.sgd

import main.scala.dataparse._

import scala.annotation.tailrec
import scala.math.{exp, pow, sqrt}

import cern.colt.matrix.impl.SparseDoubleMatrix1D
import cern.colt.matrix.linalg._



object SGD extends App {

    val training = DataSet("training")
    val eta = 0.05

 //    val iter = training.dataIterator

 //    val line: DataLine = iter.next

 //    println(line)

 //    val (features, index) = line.featuresArray
 //    features.map(println)
 //    index.map(println)
 //    val sfv = new SparseDoubleMatrix1D(25)
 //    for (i <- 0 until index.size) {
 //        println(index(i), features(i))
 //        sfv.setQuick(index(i), features(i))
 //    }
 //    println(sfv.getQuick(17010))
 //    println(sfv.toString)

 //    sfv.setQuick(10, 15.0)
 //    sfv.setQuick(25, 15.0)
 //    sfv.setQuick(24, 15.0)
 //    println(sfv.toString)

 //    // println(sfv.getQuick(17010))
 //    // println(sfv.cardinality)

 //    val sfw = sfv.copy
 //    println(sfw.getQuick(24))

	// val blas = SeqBlas.seqBlas
 //    SeqBlas.seqBlas.daxpy(1.0, sfw, sfv)

 //    println(sfv)
 //        println(sfv.getQuick(24))


    val LR = new LogisticRegression(training, eta)
    val (wFin, avgLossListFin) = LR.calculateWeights()

    // println("The L2 norm: " + l2Norm(wFin))
    // println("The avg loss list: " + avgLossListFin.length)
    // println("The avg loss first: " + avgLossListFin.head)
    // println("The avg loss last: " + avgLossListFin.last)

    training.closeData

}

class LogisticRegression(data: DataSet, eta: Double) {
    val offset = data.offset
    val maxTokenValue = data.maxTokenValue
    val tokensLength = data.tokensLength

    val blas = SeqBlas.seqBlas

    def calculateWeights(): (SparseDoubleMatrix1D, List[Double]) = {
        val avgLossList: List[Double] = List()
        val avgLoss: Double = 0.0
        val w = new SparseDoubleMatrix1D(maxTokenValue + offset + 1, tokensLength, 0.01, 0.2)

        @tailrec def recursiveSGD(iter: Iterator[DataLine],
            weights: SparseDoubleMatrix1D,
            eta: Double,
            avgLoss: Double,
            avgLossList: List[Double],
            n: Int): (SparseDoubleMatrix1D, List[Double]) = {
            if (!iter.hasNext) (weights, avgLossList)
            else {
                // A line from the data set
                val line: DataLine = iter.next
                // Create the feature vector
                val x: SparseDoubleMatrix1D = featureVector(line)
                // The actual label, clicked or not clicked
                val y: Int = line.clicked
                // The predicted label P(Y=1|X)
                val yHat: Double = predictLabel(x, weights)
                // Calculate the new weights using SGD
                val newWeights = updateWeights(y, yHat, x, weights, eta, line)
                // Calculate the average loss, the square diff between
                // actual and predicted labels
                val avgLossNew = avgLossFunc(avgLoss, y, yHat, n)
                // A list of every 100th average loss value
                val avgLossListNew = updateAvgLossList(avgLossNew, avgLossList, n)
                println(n)

                // Move onto the next line to update the weights
                recursiveSGD(iter, weights, eta, avgLossNew, avgLossListNew, n + 1)
            }
        }
        recursiveSGD(data.dataIterator, w, eta, avgLoss, avgLossList, 1)
    }

    // Return the feature vector X from the data in a line
    def featureVector(line: DataLine): SparseDoubleMatrix1D = {
        val (features, index) = line.featuresArray
        val sfv = new SparseDoubleMatrix1D(maxTokenValue + offset + 1, index.size, 0, 0.01)
        for (i <- 0 until index.size) sfv.setQuick(index(i), features(i))
        println(sfv)
        sfv
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
            x: SparseDoubleMatrix1D,
            w: SparseDoubleMatrix1D,
            eta: Double,
            line: DataLine): SparseDoubleMatrix1D = {
        val coeff: Double = (y - yHat) * eta
        // val (features, index) = line.featuresArray
        // val ft = features.map(x => coeff * x)

        blas.daxpy(coeff, x, w)
        w
    }

    // Make a prediction for the label
    def predictLabel(x: SparseDoubleMatrix1D,
            w: SparseDoubleMatrix1D): Double = {
        val ewx: Double = exp(w.zDotProduct(x))
        ewx / (1 + ewx)
    }
}
