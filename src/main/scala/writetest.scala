import scalax.io._
import scalax.io.Resource


object WriteTest extends App {
	val output: Output = 
		Resource.fromOutputStream(new java.io.FileOutputStream("daily-scala.txt"))

	val test = List(1,2,3,4,5,6,7)

	for{// create a processor (signalling the start of a batch process)
		processor <- output.outputProcessor
		// create an output object from it
		out = processor.asOutput
		}{
		// all writes to out will be on the same open output stream/channel
		out.write("first write\n")
		out.write("second write")

		for (i <- test) out.write(i + "\n")

		}
}