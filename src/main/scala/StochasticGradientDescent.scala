package main.scala.sgd

import main.scala.dataparse._

import scala.collection._
import scala.annotation.tailrec
import scala.math.{exp, sqrt, pow}

import breeze.linalg._

object SGD extends App {

    val training = DataSet("training")
    val eta = 0.05
    val lambda = 0.000

    val LR = new LogisticRegression(training, eta, lambda)
    val (weightsFinal, avgLossListFinal) = LR.calculateWeights()

    println("")
    println("The L2 norm: "         + l2Norm(weightsFinal))
    println("Max of weights: "      + weightsFinal.max)
    println("The avg loss list: "   + avgLossListFinal.length)
    println("The avg loss first: "  + avgLossListFinal.head)
    println("The avg loss 10,000: " + avgLossListFinal(10000))
    println("The avg loss second last: "   + avgLossListFinal(23356))
    println("The avg loss last: "   + avgLossListFinal.last)

    println("Position = " + weightsFinal(2))
    println("Depth = " + weightsFinal(1))
    println("Gender = " + weightsFinal(3))
    println("Age = " + weightsFinal(4))

    println("Starting test: ")
    val test = DataSet("test")
    // val weightsFinal = SparseVector.zeros[Double](1070659 + 5 + 1)


    val LRt = new LogisticRegression(test, eta, lambda)
    val (rmse1, rmse2) = LRt.predictCTR(weightsFinal)

    println("RMSE: " + rmse1)
    println("RMSE baseline: " + rmse2)

    training.closeData
    test.closeData

    def l2Norm(w: SparseVector[Double]): Double = w.norm(2)
}

class LogisticRegression(data: DataSet, eta: Double, lambda: Double) {
    val offset = data.offset
    val maxTokenValue = 1070659

    def predictCTR(weights: SparseVector[Double]): (Double, Double) = {
        val rmse = 0.0
        val rmseBaseLine = 0.0
        val baseLine = 0.03365528

        val labels = TestLabels.label

        @tailrec def predictCTRRecursive(test: Iterator[DataLine],
            labels: Iterator[Double],
            weights: SparseVector[Double],
            baseLine: Double,
            n: Int,
            rmse: Double,
            rmseBaseLine: Double): (Double, Double) = {
            if (!test.hasNext && !labels.hasNext) (sqrt(rmse / (n - 1)), sqrt(rmseBaseLine / (n - 1)))
            else {
                // A line from the test data set
                // Make sure to exclude userid = 0 or age = 0 or gender = 0
                val line = test.next

                // The actual click through rate
                val ctr = labels.next

                // The test features
                val x: SparseVector[Double] = featureVector(line)

                // The predicted label, it's a double not binary
                val yHat: Double = predictLabel(x, weights)

                // Calculate the root mean square errors
                val newrmse = rmse + pow(ctr - yHat, 2) 
                val newrmseBaseLine = rmseBaseLine + pow(ctr - baseLine, 2) // Check answer for baseline rmse
                // Hopefully will be the same, otherwise off by one errors etc

                // Move onto the next line
                predictCTRRecursive(test, labels, weights, baseLine, n + 1, newrmse, newrmseBaseLine)
            }
        }
        predictCTRRecursive(data.dataIterator, labels, weights, baseLine, 1, rmse, rmseBaseLine)
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
            if (!iter.hasNext) (weights, avgLossList.reverse)
            else {
                // A line from the data set
                // Make sure to exclude userid = 0 or age = 0 or gender = 0
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
                // 0 or 1 variable
                val y: Int = line.clicked

                // The predicted label P(Y=1|X)
                // this is a double, not binary
                val yHat: Double = predictLabel(x, delayRegWeights)

                // Calculate the average loss, the square diff between
                // actual and predicted labels
                val avgLossNew: Double = avgLossFunc(avgLoss, y, yHat, n)

                // A list of every 100th average loss value
                val avgLossListNew: List[Double] = 
                    updateAvgLossList(avgLossNew, avgLossList, n)

                // Perform the standard regularization
                val (stdRegWeights, stdAccessTimes) = 
                    performStandardReg(delayAccessTimes, x, delayRegWeights, n)

                // Calculate the new weights using SGD
                val newWeights: SparseVector[Double] = 
                    updateWeights(y, yHat, x, stdRegWeights, eta)

                // Move onto the next line to update the weights
                recursiveSGD(iter, newWeights, stdAccessTimes, avgLossNew, avgLossListNew, n + 1)
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
        for (i <- index.tail) // Don't regularize w0
            if (accessTimes(i) == 0) accessTimes(i) = n // First appearance

        accessTimes
    }

    def performDelayedReg(accessTimes: SparseVector[Int],
            x: SparseVector[Double],
            w: SparseVector[Double],
            n: Int): (SparseVector[Double], SparseVector[Int]) = {
        val coeff = 1 - (eta * lambda)
        val xIter = x.activeKeysIterator
        if (xIter.hasNext) xIter.next // Skip w0
        for (i <- xIter) {
            val t1 = accessTimes(i)
            if (t1 != 0 && n - t1 > 1) {
                w.update(i, w(i) * pow(coeff, n - t1 - 1))
                accessTimes(i) = n
            }
        }
        (w, accessTimes)
    }

    def performStandardReg(accessTimes: SparseVector[Int],
            x: SparseVector[Double],
            w: SparseVector[Double],
            n: Int): (SparseVector[Double], SparseVector[Int]) = {
        val coeff = 1 - (eta * lambda)
        val xIter = x.activeKeysIterator // Iterator over the current features
        if (xIter.hasNext) xIter.next // Skip w0
        // for (i <- xIter) {
        //     val t1 = accessTimes(i)
        //     if (t1 == n - 1) {
        //         w.update(i, w(i) * coeff)
        //         accessTimes(i) = n
        //     }
        // }
        for (i <- xIter) {
            w.update(i, w(i) * coeff)
            accessTimes(i) = n
        }
        // val temp = w(0)
        // w *= coeff
        // w(0) = temp
        (w, accessTimes)
    }

    // Update the weight vector 
    def updateWeights(y: Double, yHat: Double,
            x: SparseVector[Double],
            w: SparseVector[Double],
            eta: Double): SparseVector[Double] = {
        // Check the order of regularization and gradient 
        val grad: Double = (y - yHat) * eta
        x *= grad
        w += x
        w
    }

    // Make a prediction for the label
    def predictLabel(x: SparseVector[Double],
            w: SparseVector[Double]): Double = {
        val ewx = exp(w.dot(x))
        ewx / (1 + ewx)
    }

    // Return the average loss
    def avgLossFunc(avgLossPrev: Double, y: Double, 
            pyx: Double, n:Int): Double = {
        // Turn click probability into clicked or not clicked
        // Ask if this is true.
    	val yHat = {
    		if (pyx >= 0.5) 1
    		else 0
    	}
        ((n - 1) * avgLossPrev + pow(y - yHat, 2)) / n
    }

    // Update the average loss list with every 100th value
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


