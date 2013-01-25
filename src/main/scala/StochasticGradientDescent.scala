import main.scala.dataparse._

import scala.annotation.tailrec

import breeze.linalg._

object SGD extends App {
	println("Hello World")

    val training = DataSet("training")

    val z = training.dataIterator


 //    println(z)

 //    val (y1,y2) = z.featuresArray

	// y2.zip(y1).map(println)

 //    val x = new SparseVector(y2, y1, 141063)

 //    println(x)

    var w = SparseVector.zeros[Double](141063)

    recursiveGradient(z, w)

	@tailrec def recursiveWeightUpdates(iter: Iterator[DataLine],
    	weights: SparseVector[Double]): SparseVector[Double] = {
    	if (!iter.hasNext) weights
    	else {
    		val line = iter.next
    		val (features, ind) = line.featuresArray
    		val x = new SparseVector(ind, features, training.tokensLength)

    		val logRegExponent = x.dot(weights) // uhhh maybe not dot product

    		recursiveWeightUpdates(iter, newWeights) // problem
    	}
    }

}