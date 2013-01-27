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
    val zz = data.dataIterator

    def calculateWeights(): (SparseVector[Double], List[Double]) = {
        val avgLossList: List[Double] = List()
        val avgLoss: Double = 0.0
        val www = SparseVector.zeros[Double](maxTokenValue + offset + 1)

        // @tailrec def recursiveSGD(iter: Iterator[DataLine],
        def recursiveSGD(iter: Iterator[DataLine],
            weights: SparseVector[Double],
            eta: Double,
            avgLoss: Double,
            avgLossList: List[Double],
            n: Int,
            xtokens: Set[Int],
            wtokens: Set[Int]): (SparseVector[Double], List[Double]) = {
            if (!iter.hasNext) (weights, avgLossList)
            else {
                // A line from the data set
                val line: DataLine = iter.next
                // Create the feature vector
                val xxx: SparseVector[Double] = featureVector(line)
                // The actual label, clicked or not clicked
                val y: Int = line.clicked
                // The predicted label P(Y=1|X)
                val yHat: Double = predictLabel(xxx, weights)
                // Calculate the new weights using SGD
                val (newWeights, xtkns, wtkns): (SparseVector[Double], Set[Int], Set[Int]) = 
                    updateWeights(y, yHat, xxx, weights, eta, n, xtokens, wtokens)
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
        val (wnew,avg) = recursiveSGD(zz, www, eta, avgLoss, avgLossList, 1, Set(), Set())
        val (wnew1,avg1) = recursiveSGD(zz, wnew, eta, avgLoss, avgLossList, 1, Set(), Set())
        val (wnew2,avg2) = recursiveSGD(zz, wnew1, eta, avgLoss, avgLossList, 1, Set(), Set())
        recursiveSGD(zz, wnew2, eta, avgLoss, avgLossList, 1, Set(), Set())
    }

    // Return the feature vector X from the data in a line
    def featureVector(line: DataLine): SparseVector[Double] = {
        val (features, index) = line.featuresArray
        new SparseVector(index, features.map(_.toDouble), index.size, maxTokenValue + offset + 1) 
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
            eta: Double,
            n: Int,
            xtokens: Set[Int],
            wtokens: Set[Int]): (SparseVector[Double], Set[Int], Set[Int]) = {
        val coeff: Double = (y - yHat) * eta
        val xNew: SparseVector[Double] = x * coeff
        println(coeff)
        println(x)
        println(xNew)
        for (j <- xNew.index.sorted) {
            println(j, xNew(j))
        }
        // x.index.map(println)
        // println
        // w.index.sorted.map(println)
        // println
        println("MULTIPLICATION")
        val wNew: SparseVector[Double] = xNew + w
        println(xNew)
        println(w)
        println(wNew)

        val xtkns = xtokens ++ w.index.toSet
        val wtkns = wtokens ++ wNew.index.toSet

        // println(xNew.data.size, w.data.size, n, tkns.size)
        println(wNew.index.size, wNew.index.toSet.size, n, xtkns.size, wtkns.size)
        val wnewsorted = wNew.index.sorted
        for (i <- 0 until wNew.index.length) {
            println(wnewsorted(i), wNew(wnewsorted(i)).toString.take(5))
        }

        // println(t(0))
        // (xNew + w, tkns)
        (wNew, xtkns, wtkns)
    }

    // Make a prediction for the label
    def predictLabel(x: SparseVector[Double],
            w: SparseVector[Double]): Double = {
        val logisticRegressExp = exp(w.dot(x))
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