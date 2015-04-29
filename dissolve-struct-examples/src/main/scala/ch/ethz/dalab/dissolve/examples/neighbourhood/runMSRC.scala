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
import breeze.linalg.Vector
import breeze.linalg.DenseMatrix
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import sys.process._
 
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
    //
    //
    
    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")
    val runLocally: Boolean = options.getOrElse("local", "true").toBoolean
    val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val gitV = ("git rev-parse HEAD"!!).replaceAll("""(?m)\s+$""", "")
    val experimentName:String = options.getOrElse("runName", "UnNamed")
    
    val msrcDir: String = "../data/generated"

    val appName: String = "ImageSegGraph"

    val printImages: Boolean = options.getOrElse("printImages","false").toBoolean
    
    val solverOptions: SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] = new SolverOptions()
    solverOptions.gitVersion = gitV
    solverOptions.runName = experimentName
    solverOptions.startTime = System.currentTimeMillis
    solverOptions.roundLimit = options.getOrElse("roundLimit", "5").toInt // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean
    solverOptions.lambda = options.getOrElse("lambda", "0.01").toDouble
    solverOptions.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    solverOptions.doLineSearch = options.getOrElse("linesearch", "true").toBoolean
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean
    solverOptions.onlyUnary = options.getOrElse("onlyUnary", "false").toBoolean
    GraphSegmentation.DISABLE_PAIRWISE = solverOptions.onlyUnary
    
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
      solverOptions.sampleFrac = 0.2
      solverOptions.enableOracleCache = false
      solverOptions.oracleCacheSize = 100
      solverOptions.stoppingCriterion = RoundLimitCriterion
      solverOptions.roundLimit = 1
      solverOptions.enableManualPartitionSize = true
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false
      solverOptions.debug = true
      solverOptions.debugMultiplier = 1
    }
    solverOptions.numClasses = 24
    // (Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]], Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]]) 

    val (oldtrainData, oldtestData) = ImageSegmentationUtils.loadMSRC("../data/generated/MSRC_ObjCategImageDatabase_v2")

    val graphTrainD = for (i <- 0 until oldtrainData.size) yield {
      val (gTrain, gLabel) = GraphUtils.convertOT_msrc_toGraph(oldtrainData(i).pattern, oldtrainData(i).label, solverOptions.numClasses)
      new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](gLabel, gTrain)
    }
    val graphTestD = for (i <- 0 until oldtestData.size) yield {
      val (gTrain, gLabel) = GraphUtils.convertOT_msrc_toGraph(oldtestData(i).pattern, oldtestData(i).label, solverOptions.numClasses)
      new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](gLabel, gTrain)
    }
    val trainData = graphTrainD.toArray.toSeq
    val testData = graphTestD.toArray.toSeq

    //TODO remove this debug (this could be made into a testcase ) 
    if(false){
    val compAll = for ( i <- 0 until 10) yield{
    val first = trainData(i)
    val old = oldtrainData(i).label

    val re = GraphUtils.reConstruct3dMat(first.label, first.pattern.dataGraphLink,
      first.pattern.maxCoord._1+1,
      first.pattern.maxCoord._2+1, first.pattern.maxCoord._3+1)
    val flatRe = GraphUtils.flatten3rdDim(re)
    
    def compareROIl( left : DenseMatrix[ROILabel], right : Array[Array[Int]] ):Int={
      var counter =0;
      for( r <- 0 until left.rows){
        for( c <- 0 until left.cols){
          if(left(r,c).label!=right(r)(c))
            counter +=1
        }
      }
      return counter
    }
    val tmp = compareROIl(old,flatRe)
    print("dif "+ tmp)
    //Now lets print the pictures 
    GraphUtils.printBMPfrom3dMat(flatRe,"reconst_"+i+"_.bmp");
    
    //
    tmp
    }
    }
      
    //

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
        Some(sc.parallelize(trainData, solverOptions.NUM_PART))
      else
        Some(sc.parallelize(trainData))

    val trainDataRDD =
      if (solverOptions.enableManualPartitionSize)
        sc.parallelize(trainData, solverOptions.NUM_PART)
      else
        sc.parallelize(trainData)

    val trainer: StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] =
      new StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels](
        trainDataRDD,
        GraphSegmentation,
        solverOptions)

    val model: StructSVMModel[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] = trainer.trainModel()

    var avgTrainLoss = 0.0

    var count=0
    for (item <- trainData) {
      val prediction = model.predict(item.pattern)
      
      if(printImages){
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(item.label, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Train"+count+"true.bmp")
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(prediction, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Train"+count+"pred.bmp")
      }
      avgTrainLoss += GraphSegmentation.lossFn(item.label, prediction)
      count+=1
    }
    avgTrainLoss = avgTrainLoss / trainData.size
    println("\nTRAINING: Avg Loss : " + avgTrainLoss + " numItems " + testData.size)
    //Test Error 
    avgTrainLoss = 0.0
    count=0
    for (item <- testData) {
      val prediction = model.predict(item.pattern)
      
      if(printImages){
            GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(item.label, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"imgTest"+count+"trueRW.bmp")
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(prediction, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"imgTest"+count+"predRW.bmp")
      }
      avgTrainLoss += GraphSegmentation.lossFn(item.label, prediction)
      count+=1
    }
    avgTrainLoss = avgTrainLoss / testData.size
    println("\nTest Avg Loss : " + avgTrainLoss + " numItems " + testData.size)
    sc.stop()

  }

}