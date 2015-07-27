package ch.ethz.dalab.dissolve.diagnostics

import java.nio.file.Paths

import org.scalatest.FlatSpec
import org.scalatest.Inside
import org.scalatest.Inspectors
import org.scalatest.Matchers
import org.scalatest.OptionValues
import breeze.linalg.DenseVector
import breeze.linalg.Matrix
import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.examples.chain.ChainDemo
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg.max
import ch.ethz.dalab.dissolve.examples.neighbourhood.GraphLabels
import ch.ethz.dalab.dissolve.examples.neighbourhood.GraphStruct
import ch.ethz.dalab.dissolve.examples.neighbourhood.GraphSegmentationClass
import ch.ethz.dalab.dissolve.examples.neighbourhood.GraphSegmentationClass
import ij._
import ij.io.Opener
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import org.apache.spark.SparkConf
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import org.apache.spark.SparkContext


 
object ChainTestAdapter_G {

  ////

  type X = GraphStruct[Vector[Double], (Int, Int, Int)]
  type Y = GraphLabels

  val myGraphSegObj = new GraphSegmentationClass(true, MAX_DECODE_ITERATIONS=1000, MAX_DECODE_ITERATIONS_MF_ALT = 10, USE_NAIV_UNARY_MAX=false,USE_MF=false, DISABLE_UNARY = false)

  val dissolveFunctions: DissolveFunctions[X, Y] = myGraphSegObj

  val histBinsPerCol = 3
  val histBinsPerGray = 8
  import ch.ethz.dalab.dissolve.examples.neighbourhood.runMSRC._
  val featFn2 = (image: ImageStack, mask: Array[Array[Array[Int]]]) => {

    val xDim = mask.length
    val yDim = mask(0).length
    val zDim = mask(0)(0).length
    val numSupPix = mask(xDim - 1)(yDim - 1)(zDim - 1) + 5 //TODO is this correct always ?
   
    val isColor =  true //TODO maybe this is not the best way to check for color in the image
    //TODO the bit depth should give me the max value which the hist should span over 

    if (isColor) {
      val hist = colorhist(image, mask, histBinsPerCol, 255 / histBinsPerCol)
     // val coMat = coOccurancePerSuperRGB(mask, image, numSupPix, histBinsPerCol)
      hist///++ coMat
    } else {
      val hist = greyHist(image, mask, histBinsPerGray, 255 / (histBinsPerGray))
      val coMat= greyCoOccurancePerSuper(image, mask, histBinsPerCol)
      hist ++coMat
    }
  }
  
  val dataPath =  "/home/mort/workspace/dissolve-struct/data/generated/colorEasy"
  val numClasses=2
  println (" Using this data: "+ dataPath)
  //TODO add features to this noise creator which makes groundTruth files just like those in getMSRC or getMSRCSupPix
  val (trainData, testData, colorlabelMap, classFreqFound, transProb) = genMSRCsupPixV2(2, 10,10,dataPath+"/Images", dataPath+"/GroundTruth", featFn2, 100, "none8", false, false, false)


  val data = trainData.toArray
  
 
   val lo = data(0)
  val numd = myGraphSegObj.featureFn(lo.pattern, lo.label).size
  
  val model: StructSVMModel[X, Y] =
    new StructSVMModel[X, Y](DenseVector.zeros(numd), 0.0,
      DenseVector.zeros(numd), dissolveFunctions, numClasses) // T

      
      
      /**
   * Perturb
   * Return a compatible perturbed Y
   * Higher the degree, more perturbed y is
   *
   * This function perturbs `degree` of the values by swapping
   */
  def perturb(y: Y, degree: Double = 0.1): Y = {
    val d = y.d.size
    val numSwaps = max(1, (degree * d).toInt)

    for (swapNo <- 0 until numSwaps) {
      // Swap two random values in y
      val (i, j) = (scala.util.Random.nextInt(d), scala.util.Random.nextInt(d))
      val temp = y.d(i)
      y.d(i) = y.d(j)
      y.d(j) = temp
    }

    y

  }
}


object ChainTestAdapter {
  type X = Matrix[Double]
  type Y = Vector[Double]

  /**
   * Dissolve Functions
   */
  val dissolveFunctions: DissolveFunctions[X, Y] = ChainDemo
  /**
   * Some Data
   */
  val data = {
    val dataDir = "../data/generated"
    val trainDataSeq: Vector[LabeledObject[Matrix[Double], Vector[Double]]] =
      ChainDemo.loadData(dataDir + "/patterns_train.csv",
        dataDir + "/labels_train.csv",
        dataDir + "/folds_train.csv")

    trainDataSeq.toArray
  }
  /**
   * A dummy model
   */
  val lo = data(0)
  val numd = ChainDemo.featureFn(lo.pattern, lo.label).size
  val model: StructSVMModel[X, Y] =
    new StructSVMModel[X, Y](DenseVector.zeros(numd), 0.0,
      DenseVector.zeros(numd), dissolveFunctions, 1)

  /**
   * Perturb
   * Return a compatible perturbed Y
   * Higher the degree, more perturbed y is
   *
   * This function perturbs `degree` of the values by swapping
   */
  def perturb(y: Y, degree: Double = 0.1): Y = {
    val d = y.size
    val numSwaps = max(1, (degree * d).toInt)

    for (swapNo <- 0 until numSwaps) {
      // Swap two random values in y
      val (i, j) = (scala.util.Random.nextInt(d), scala.util.Random.nextInt(d))
      val temp = y(i)
      y(i) = y(j)
      y(j) = temp
    }

    y

  }
}

/**
 * @author torekond
 */
abstract class UnitSpec extends FlatSpec with Matchers with OptionValues with Inside with Inspectors {

  val DissolveAdapter = ChainTestAdapter_G

  type X = DissolveAdapter.X
  type Y = DissolveAdapter.Y

  val dissolveFunctions = DissolveAdapter.dissolveFunctions
  val data = DissolveAdapter.data
  val model = DissolveAdapter.model

  /**
   * Helper functions
   */
  def perturb = DissolveAdapter.perturb _

  // Joint Feature Map
  def phi = dissolveFunctions.featureFn _
  def delta = dissolveFunctions.lossFn _
  def maxoracle = dissolveFunctions.oracleFn _

  def psi(lo: LabeledObject[X, Y], ymap: Y) =
    phi(lo.pattern, lo.label) - phi(lo.pattern, ymap)

  def F(x: X, y: Y, w: Vector[Double]) =
    w dot phi(x, y)
  def deltaF(lo: LabeledObject[X, Y], ystar: Y, w: Vector[Double]) =
    w dot psi(lo, ystar)

}

