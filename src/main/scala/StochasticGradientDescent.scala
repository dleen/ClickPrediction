package main.scala.sgd

import main.scala.dataparse._

import scala.annotation.tailrec
import scala.math._

import breeze.linalg._

import cern.colt.matrix.impl.SparseDoubleMatrix1D
import cern.colt.matrix.linalg.Blas._

object SGD extends App {

    val training = DataSet("training")
    val eta = 0.05

    val iter = training.dataIterator

    val line: DataLine = iter.next

    println(line)

    val (features, index) = line.featuresArray
    features.map(println)
    index.map(println)
    val sfv = new SparseDoubleMatrix1D(1)
    for (i <- 0 until index.size) {
        println(index(i), features(i))
        sfv.setQuick(index(i), features(i))
    }
    println(sfv.getQuick(17010))
    println(sfv.toString)

    sfv.setQuick(10, 15.0)
    sfv.setQuick(25, 15.0)
    sfv.setQuick(24, 15.0)

    println(sfv.getQuick(17010))
    println(sfv.cardinality)

    val sfw = sfv.copy

    daxpy(1.0, sfw, sfv)

    // val LR = new LogisticRegression(training, eta)
    // val (wFin, avgLossListFin) = LR.calculateWeights()

    // println("The L2 norm: " + l2Norm(wFin))
    // println("The avg loss list: " + avgLossListFin.length)
    // println("The avg loss first: " + avgLossListFin.head)
    // println("The avg loss last: " + avgLossListFin.last)

    training.closeData

    def l2Norm(w: SparseVector[Double]): Double = {
        sqrt(w.dot(w))
    }    
}

class LogisticRegression(data: DataSet, eta: Double) {
    val offset = data.offset
    val maxTokenValue = data.maxTokenValue
    val tokensLength = data.tokensLength
    val zz = data.dataIterator

    def calculateWeights(): (SparseDoubleMatrix1D, List[Double]) = {
        val avgLossList: List[Double] = List()
        val avgLoss: Double = 0.0
        val w = new SparseDoubleMatrix1D(tokensLength + offset + 1)

        // @tailrec def recursiveSGD(iter: Iterator[DataLine],
        def recursiveSGD(iter: Iterator[DataLine],
            weights: SparseDoubleMatrix1D,
            eta: Double,
            avgLoss: Double,
            avgLossList: List[Double],
            n: Int,
            xtokens: Set[Int],
            wtokens: Set[Int]): (SparseDoubleMatrix1D, List[Double]) = {
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
                val (newWeights, xtkns, wtkns) = 
                    updateWeights(y, yHat, x, weights, eta, n, xtokens, wtokens)
                // Calculate the average loss, the square diff between
                // actual and predicted labels
                val avgLossNew = avgLossFunc(avgLoss, y, yHat, n)
                // A list of every 100th average loss value
                val avgLossListNew = updateAvgLossList(avgLossNew, avgLossList, n)

                // Move onto the next line to update the weights
                (newWeights, avgLossListNew)
                //recursiveSGD(iter, newWeights, eta, avgLossNew, avgLossListNew, n + 1, xtkns, wtkns)
            }
        }
        val (wnew,avg) = recursiveSGD(zz, w, eta, avgLoss, avgLossList, 1, Set(), Set())
        val (wnew1,avg1) = recursiveSGD(zz, wnew, eta, avgLoss, avgLossList, 1, Set(), Set())
        val (wnew2,avg2) = recursiveSGD(zz, wnew1, eta, avgLoss, avgLossList, 1, Set(), Set())
        recursiveSGD(zz, wnew2, eta, avgLoss, avgLossList, 1, Set(), Set())
    }

    // Return the feature vector X from the data in a line
    def featureVector(line: DataLine): SparseDoubleMatrix1D = {
        val (features, index) = line.featuresArray
        val sfv = new SparseDoubleMatrix1D(index.size)
        for (i <- 0 until index.size) sfv.setQuick(index(i), features(i))
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
            n: Int,
            xtokens: Set[Int],
            wtokens: Set[Int]): (SparseDoubleMatrix1D, Set[Int], Set[Int]) = {
        // val coeff: Double = (y - yHat) * eta
        // val xNew: SparseDoubleMatrix1D = x * coeff

        // // x.index.map(println)
        // // println
        // // w.index.sorted.map(println)
        // // println
        // val wNew: SparseDoubleMatrix1D = xNew + w
        // println(xNew)
        // println(w)
        // println(wNew)

        // val xtkns = xtokens ++ w.index.toSet
        // val wtkns = wtokens ++ wNew.index.toSet



        (w, xtokens, wtokens)
    }

    // Make a prediction for the label
    def predictLabel(x: SparseDoubleMatrix1D,
            w: SparseDoubleMatrix1D): Double = {
        val logisticRegressExp: Double = exp(w.zDotProduct(x))
        logisticRegressExp / (1 + logisticRegressExp)
    }
}

object smalltest extends App {
    val w = new SparseVector(Array(0,1,2),Array(0,1,2),6)
    val z = new SparseVector(Array(0,1,5),Array(3,4,5),6)
    val t = w + z * 2 * 1
    


    (t + t).index.map(println)
    // w.map(println)

}