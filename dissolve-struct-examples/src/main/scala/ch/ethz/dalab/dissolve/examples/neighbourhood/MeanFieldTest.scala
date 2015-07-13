package ch.ethz.dalab.dissolve.examples.neighbourhood

import breeze.linalg._
import breeze.stats.DescriptiveStats._
import breeze.stats._

object MeanFieldTest {
 
   def decodeFn_PR(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], connections: Array[Array[Int]], maxIterations: Int, debug: Boolean = false, temp: Double = 5, logTag: String = "NoTag"): Array[Int] = {
     
     val DISABLE_PAIRWISE = if (thetaPairwise.size == 0) true else false
    // val probTheta = (thetaPairwise-thetaMin)/(thetaMax-thetaMin) //TODO conforming theta to be a probability like this is pretty bad. I dont think we can solve a non probabiltiy with mean field 

    val t0 = System.currentTimeMillis()

    val numRegions: Int = thetaUnary.rows
    assert(numRegions == connections.size)
    val numClasses: Int = thetaUnary.cols

    val Q = DenseMatrix.ones[Double](numRegions, numClasses) //TODO maybe initialize to 1/num neighbours or 1/numneighbours*numClass
    //TODO This Q could be stored in a spark RDD if it gets too large for memory;; Prob not necesary because thetaUnayr is the same size 
    //TODO consider storing the probabilities in a byte
    if (debug) {
      //val header = (0 until numClasses).toSeq.toList.flatMap { curclass => (0 until numRegions).toSeq.toList.map { curregion => ",Qr" + curregion + " Qc" + curclass } }
      // println("#MF_Q_LOG#,timeID,elapsedTime" + header)
      // println("#MF_E_LOG#,timeID,logTag,iter,maxE,minE,maxE_change,minE_change")
    }

    var lastMaxE = 0.0
    var lastMinE = 0.0
    var numNoChange = 0
    for (iter <- 0 until maxIterations) {

      var numUnchangedQs = 0
      val lastQ = Q;
      val xiLab = (0 until numClasses).par
      //val xis = (0 until graph.size).par

      val allXiperLabel = xiLab.map(curLab => ((curLab,
        for (xi <- 0 until connections.size) yield {

          val neigh = connections(xi)
          val allClasses = (0 until numClasses).toList

          val newQest = neigh.toList.map { neighIdx =>
            allClasses.foldLeft(0.0) { (running, curClass) =>
              {
                running + Math.exp(lastQ(neighIdx, curClass) * (if (DISABLE_PAIRWISE) 0 else thetaPairwise(curClass, curLab))) * Math.exp((1 / temp) * thetaUnary(xi, curLab))
              }
            }
          }.sum

          (1 / temp) * newQest
        })))

      for (labAgain <- 0 until numClasses) {
        val allXi = allXiperLabel(labAgain)._2.toArray
        Q(::, labAgain) := DenseVector(allXi)
      }
      //allXiperLabel.foreach((label:Int,listofQs:IndexedSeq[Double])=>Q(::,label):=DenseVector(listofQs.toArray))

      if (debug) {
        //     println("#MF_Q_LOG#, %d,%d".format(t0, System.currentTimeMillis() - t0) + Q.toDenseVector.toArray.mkString("", ",", ""))
        val debugE = Vector(Array.fill(numRegions)(0))
        val allMaxQ = for (row <- 0 until Q.rows) yield {
          max(Q(row, ::))
        }
        val allMinQ = for (row <- 0 until Q.rows) yield {
          min(Q(row, ::))
        }
        val tmp = allMaxQ.product
        val maxQ = allMaxQ.product
        val minQ = allMinQ.product
        println("#MF_E_LOG#,%d,%s,%d,%e,%e,%e,%e".format(t0, logTag, iter, maxQ, minQ, maxQ - lastMaxE, minQ - lastMinE))
        if (lastMaxE - maxQ == 0.0 && lastMinE - minQ == 0.0)
          numNoChange += 1
        lastMaxE = maxQ
        lastMinE = minQ

        if (numNoChange >= 2) {

          val outLabt = Array.fill(numRegions)(0)
          for (row <- 0 until Q.rows) {
            outLabt(row) = argmax(Q(row, ::))
          }
          return outLabt
        }
      }

    }
    val outLab = Array.fill(numRegions)(0)
    for (row <- 0 until Q.rows) {
      outLab(row) = argmax(Q(row, ::))
    }

    val t1 = System.currentTimeMillis()
    // print("[MF decodeTime=[%d s] ]".format(  (t1-t0)/1000  ))
    return outLab
   
   }
  

  
}