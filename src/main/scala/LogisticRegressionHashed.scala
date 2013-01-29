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