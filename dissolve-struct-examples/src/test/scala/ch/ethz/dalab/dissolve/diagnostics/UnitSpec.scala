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
import ch.ethz.dalab.dissolve.examples.neighbourhood.startupUtils._

 
object ChainTestAdapter_G {

  final def sample[A](dist: Map[A, Double]): A = {
  val p = scala.util.Random.nextDouble
  val it = dist.iterator
  var accum = 0.0
  while (it.hasNext) {
    val (item, itemProb) = it.next
    accum += itemProb
    if (accum >= p)
      return item  // return so that we don't have to search through the whole distribution
  }
  sys.error(f"this should never happen")  // needed so it will compile
}
  
  ////
  val dataPath =  "/home/mort/workspace/dissolve-struct/data/generated/Mito_2d"
  val numClasses=2
  

  type X = GraphStruct[Vector[Double], (Int, Int, Int)]
  type Y = GraphLabels
 val sO: SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] = new SolverOptions()
  sO.recompFeat=false
  sO.onlyUnary=false
  sO.useNaiveUnaryMax=true
  sO.useLoopyBP=true
  sO.useMF=false
  sO.loopyBPmaxIter=2
  sO.alsoWeighLossAugByFreq=true
  sO.useClassFreqWeighting=true
  sO.isColor=false
  sO.generateMSRCSupPix=false
  sO.squareSLICoption=true 
  sO.superPixelSize=15  //S
  sO.slicCompactness=(-1)
  sO.featHistSize=15
  sO.featIncludeMeanIntensity = true
  sO.featAddOffsetColumn=false
  sO.featAddIntensityVariance=false
  sO.featNeighHist=true
  sO.featUnique2Hop=false
  sO.featUniqueIntensity = false
  sO.slicNormalizePerClust=false  
  sO.standardizeFeaturesByColumn=true
  sO.numClasses=numClasses
  sO.runName="UnNamed"

  sO.imageDataFilesDir=dataPath+"/Images"
    sO.groundTruthDataFilesDir=dataPath+"/GroundTruth"
   
   import ch.ethz.dalab.dissolve.examples.neighbourhood.runReadTrainPredict._
  
  
  
  println (" Using this data: "+ dataPath)
  //TODO add features to this noise creator which makes groundTruth files just like those in getMSRC or getMSRCSupPix
  
  val (trainData,testData, colorlabelMap, classFreqFound,transProb, newSo) = genGraphFromImages(sO,featFn3,afterFeatFn1)

  val data = trainData.toArray
    
  val myGraphSegObj = new GraphSegmentationClass(sO.onlyUnary,1000,
            sO.learningRate ,sO.useMF,sO.mfTemp,sO.useNaiveUnaryMax,
            false,10,sO.runName,
             classFreqFound,
            sO.weighDownUnary,sO.weighDownPairwise, sO.LOSS_AUGMENTATION_OVERRIDE,
            false,sO.PAIRWISE_UPPER_TRI,sO.useMPLP,sO.useLoopyBP,sO.loopyBPmaxIter,sO.alsoWeighLossAugByFreq,sO) 
  val dissolveFunctions: DissolveFunctions[X, Y] = myGraphSegObj

 
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
  def perturb(y: Y, degree: Double = 0.3): Y = {
    val d = y.d.size
    val numSwaps = max(1, (degree * d).toInt)

    val possibleIDX =scala.util.Random.shuffle((0 until d).toList)
    for (swapNo <- 0 until numSwaps) {
      // Swap two random values in y
      val nextNewLabel =sample[Int](classFreqFound)
      val i = possibleIDX(swapNo)
      
      y.d(i) = nextNewLabel
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

