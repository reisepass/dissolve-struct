package ch.ethz.dal.img3d.test

import org.scalatest._
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.examples.imgseg3d.ImageSeg3d
import ch.ethz.dalab.dissolve.optimization.RoundLimitCriterion
import ch.ethz.dalab.dissolve.examples.imgseg3d.ThreeDimMat
import ch.ethz.dalab.dissolve.examples.imgseg3d.NominalThreeDimMat
import ch.ethz.dalab.dissolve.examples.imgseg3d.ThreeDimUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import ch.ethz.dalab.dissolve.classification.StructSVMModel

class Img3dSegSanityCheck extends FlatSpec {

  behavior of "the first Y* (most constraining y returned from the oracle)"

  val dataDir: String = "../data/generated"
  val debugDir: String = "../debug"
  val runLocally: Boolean = true
  val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)
  val msrcDir: String = "../data/generated"
  val appName: String = "ImageSeg3d"
  val solverOptions: SolverOptions[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]] = new SolverOptions()
  solverOptions.sampleFrac = 0.5
  solverOptions.enableOracleCache = false
  solverOptions.oracleCacheSize = 100
  solverOptions.stoppingCriterion = RoundLimitCriterion
  solverOptions.roundLimit = 5
  solverOptions.enableManualPartitionSize = true
  solverOptions.NUM_PART = 1
  solverOptions.doWeightedAveraging = false
  solverOptions.debug = false
  solverOptions.debugMultiplier = 1
  solverOptions.numClasses = 2
  val (trainData, testData) = ThreeDimUtils.generateSomeData(20, 25, 3, 5, 2)

  it should "be the inverse of the true Y if w initialized as a vector of zeros " in {
    val conf = new SparkConf().setAppName(appName).setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(debugDir + "/checkpoint-files")
    solverOptions.testDataRDD = Some(sc.parallelize(testData.toSeq, solverOptions.NUM_PART))
    val trainDataRDD = sc.parallelize(trainData, solverOptions.NUM_PART)
    val trainer: StructSVMWithDBCFW[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]] =
      new StructSVMWithDBCFW[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]](
        trainDataRDD,
        ImageSeg3d,
        solverOptions)

    val model: StructSVMModel[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]] = trainer.trainModel()
    //TODO idk how to get the W out after one iteration 
    //Maybe i can add some Debug stuff that will output a history of Y* maxOracle outputs 

  }

}