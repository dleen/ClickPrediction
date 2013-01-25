package main.scala.sgd

import main.scala.dataparse._

import scala.annotation.tailrec
import scala.math._

import breeze.linalg._

object SGD extends App {

    val training = DataSet("training")
    val z = training.dataIterator

    val w = SparseVector.zeros[Double](training.maxTokenValue + training.offset + 1)

    val eta = 0.05

    val averageLoss: List[Double] = List()

    recursiveWeightUpdates(z, w)

    @tailrec def recursiveSGD(iter: Iterator[DataLine],
        weights: SparseVector[Double], averageLoss: List[Double], n: Int): SparseVector[Double] = {
        if (!iter.hasNext) weights
        else {
            val line = iter.next
            val (features, index) = line.featuresArray
            val x = new SparseVector(index, features, training.maxTokenValue + training.offset + 1)

            val y = line.clicked
            val yHat = predictLabel(x, weights)

            val newWeights = updateWeights(y, yHat, x, w)

            if (n % 100 == 0) {
                val avgLossNew = averageLossFunc(averageLoss, y, yHat, n)
                val avgLoss = avgLossNew :: averageLoss
            } else {
                val avgLoss = averageLoss
            }

            recursiveSGD(iter, newWeights, avgLoss) // problem
        }
    }

    def averageLossFunc(avgLossPrev: Double, actualLabel: Double, 
        predictedLabel: Double, n:Int): Double = {
        ((n - 1) * avgLossPrev + pow(actualLabel - predictedLabel, 2)) / n
    }

    def updateWeights(actualLabel: Double, 
        predictedLabel: Double, x: SparseVector[Double],
        w: SparseVector[Double]): SparseVector[Double] = {
        w + x * (actualLabel - predictedLabel) * eta 
    }

    def predictLabel(x: SparseVector[Double],
        w: SparseVector[Double]): Double = {
        val logisticRegressExp = exp(x.dot(w))
        logisticRegressExp / (1 + logisticRegressExp)
    }

}