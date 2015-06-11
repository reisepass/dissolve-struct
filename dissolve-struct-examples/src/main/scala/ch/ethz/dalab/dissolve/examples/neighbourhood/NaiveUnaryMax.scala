package ch.ethz.dalab.dissolve.examples.neighbourhood


import breeze.linalg._
import breeze.stats.DescriptiveStats._
import breeze.stats._


object NaiveUnaryMax {
type xData = GraphStruct[Vector[Double], (Int, Int, Int)]


def decodeFn(thetaUnary: DenseMatrix[Double]): GraphLabels = {

   
    
    

    val numRegions: Int = thetaUnary.rows
    val numClasses: Int = thetaUnary.cols
  
   val outLab = Vector(Array.fill(numRegions)(0))
    for (row <- 0 until thetaUnary.rows) {
      outLab(row)=argmax(thetaUnary(row,::))
    }
    return ( new GraphLabels(outLab,numClasses))
}
}