package ch.ethz.dalab.dissolve.examples.neighbourhood

import ch.ethz.dalab.dissolve.examples.imageseg._
import org.apache.spark.SparkConf
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import org.apache.spark.SparkContext
import ch.ethz.dalab.dissolve.examples.imgseg3d.ThreeDimUtils
import org.apache.log4j.PropertyConfigurator
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import ch.ethz.dalab.dissolve.examples.neighbourhood._
import ch.ethz.dalab.dissolve.optimization.RoundLimitCriterion
import ch.ethz.dalab.dissolve.regression.LabeledObject

object runMSRC {
  
  
   def main(args: Array[String]): Unit = {

    PropertyConfigurator.configure("conf/log4j.properties")

    val options: Map[String, String] = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt)    => (opt -> "true")
        case _             => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    System.setProperty("spark.akka.frameSize", "512")
    println(options)
    runStuff(options)
  }
  def runStuff(options: Map[String, String]) {

    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")
    val runLocally: Boolean = options.getOrElse("local", "true").toBoolean
    val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val msrcDir: String = "../data/generated"

    val appName: String = "ImageSegGraph"

    val solverOptions: SolverOptions[GraphStruct[Vector[Double], (Int,Int,Int)],GraphLabels] = new SolverOptions()
    solverOptions.roundLimit = options.getOrElse("roundLimit", "5").toInt // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean
    solverOptions.lambda = options.getOrElse("lambda", "0.01").toDouble
    solverOptions.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    solverOptions.doLineSearch = options.getOrElse("linesearch", "true").toBoolean
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean

    solverOptions.sample = options.getOrElse("sample", "frac")
    solverOptions.sampleFrac = options.getOrElse("samplefrac", "0.5").toDouble
    solverOptions.sampleWithReplacement = options.getOrElse("samplewithreplacement", "false").toBoolean

    solverOptions.enableManualPartitionSize = options.getOrElse("manualrddpart", "false").toBoolean
    solverOptions.NUM_PART = options.getOrElse("numpart", "2").toInt

    solverOptions.enableOracleCache = options.getOrElse("enableoracle", "false").toBoolean
    solverOptions.oracleCacheSize = options.getOrElse("oraclesize", "5").toInt

    solverOptions.debugInfoPath = options.getOrElse("debugpath", debugDir + "/imageseg-%d.csv".format(System.currentTimeMillis()))
    /**
     * Some local overrides
     */
    if (runLocally) {
      solverOptions.sampleFrac = 0.5
      solverOptions.enableOracleCache = false
      solverOptions.oracleCacheSize = 100
      solverOptions.stoppingCriterion = RoundLimitCriterion
      solverOptions.roundLimit = 5
      solverOptions.enableManualPartitionSize = true
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false
      solverOptions.debug = true
      solverOptions.debugMultiplier = 1
    }

    // (Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]], Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]]) 
   
     val (trainData, testData) = ImageSegmentationUtils.loadMSRC("../data/generated/MSRC_ObjCategImageDatabase_v2")
     
     val graphTrainD = for ( i <- 0 until trainData.size) yield{ 
        val ( gTrain, gLabel) =  GraphUtils.convertOT_msrc_toGraph(trainData(i).pattern, trainData(i).label,solverOptions.numClasses)
        new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int,Int,Int)], GraphLabels](gLabel,gTrain)
     }
     val graphTestD = for ( i <- 0 until testData.size) yield{
        val ( gTrain, gLabel) = GraphUtils.convertOT_msrc_toGraph(testData(i).pattern, testData(i).label,solverOptions.numClasses)
        new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int,Int,Int)], GraphLabels](gLabel,gTrain)
     }
    
    //BOOKMARK  fix the blow errors by exchanging all the trainData -> graphTrainD
    solverOptions.numClasses = 2

    println(solverOptions.toString())

    val conf =
      if (runLocally)
        new SparkConf().setAppName(appName).setMaster("local")
      else
        new SparkConf().setAppName(appName)

    val sc = new SparkContext(conf)
    sc.setCheckpointDir(debugDir + "/checkpoint-files")

    println(SolverUtils.getSparkConfString(sc.getConf))

    solverOptions.testDataRDD =
      if (solverOptions.enableManualPartitionSize)
        Some(sc.parallelize(testData.toSeq, solverOptions.NUM_PART))
      else
        Some(sc.parallelize(testData.toSeq))

    val trainDataRDD =
      if (solverOptions.enableManualPartitionSize)
        sc.parallelize(trainData, solverOptions.NUM_PART)
      else
        sc.parallelize(trainData)

    val trainer: StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int,Int,Int)],GraphLabels] =
      new StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int,Int,Int)],GraphLabels](
        trainDataRDD,
        GraphSegmentation, 
        solverOptions)

    val model: StructSVMModel[GraphStruct[Vector[Double], (Int,Int,Int)],GraphLabels] = trainer.trainModel()

    var avgTrainLoss = 0.0
   

    for (item <- trainData) {
      val prediction = model.predict(item.pattern)
      avgTrainLoss += lossFn(item.label, prediction)
    }
    avgTrainLoss = avgTrainLoss / trainData.size
    println("\nTRAINING: Avg Loss : " + avgTrainLoss + " numItems " + testData.size)
    //Test Error 
    avgTrainLoss = 0.0
     for (item <- testData) {
      val prediction = model.predict(item.pattern)
      avgTrainLoss += lossFn(item.label, prediction)
    }
    avgTrainLoss = avgTrainLoss / testData.size
    println("\nTest Avg Loss : " + avgTrainLoss + " numItems " + testData.size)

  }
  
  
 

}