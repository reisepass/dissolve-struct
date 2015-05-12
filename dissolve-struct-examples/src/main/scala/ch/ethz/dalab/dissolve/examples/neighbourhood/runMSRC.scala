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
    //TODO Remove debug
    //( canvasSize : Int,probUnifRandom: Double, featureNoise : Double, pairRandomItr: Int, numClasses:Int, neighbouringProb : Array[Array[Double]], classFeat:Array[Vector[Double]]){

    //
    

    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")
    val runLocally: Boolean = options.getOrElse("local", "false").toBoolean
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
    //GraphSegmentation.DISABLE_PAIRWISE = solverOptions.onlyUnary
    val MAX_DECODE_ITERATIONS:Int = options.getOrElse("maxDecodeItr",  (if(solverOptions.onlyUnary) 100 else 1000 ).toString ).toInt
    
    solverOptions.sample = options.getOrElse("sample", "frac")
    solverOptions.sampleFrac = options.getOrElse("samplefrac", "0.5").toDouble
    solverOptions.sampleWithReplacement = options.getOrElse("samplewithreplacement", "false").toBoolean

    solverOptions.enableManualPartitionSize = options.getOrElse("manualrddpart", "false").toBoolean
    solverOptions.NUM_PART = options.getOrElse("numpart", "2").toInt

    solverOptions.enableOracleCache = options.getOrElse("enableoracle", "false").toBoolean
    solverOptions.oracleCacheSize = options.getOrElse("oraclesize", "5").toInt

    solverOptions.useMF=options.getOrElse("useMF","false").toBoolean
    solverOptions.useNaiveUnaryMax = options.getOrElse("useNaiveUnaryMax","false").toBoolean
    solverOptions.learningRate = options.getOrElse("learningRate","0.1").toDouble
    solverOptions.mfTemp=options.getOrElse("mfTemp","5.0").toDouble
    solverOptions.debugInfoPath = options.getOrElse("debugpath", debugDir + "/imageseg-%d.csv".format(System.currentTimeMillis()))
    solverOptions.useMSRC = options.getOrElse("useMSRC","false").toBoolean
    solverOptions.dataGenSparsity = options.getOrElse("dataGenSparsity","-1").toDouble
    solverOptions.dataAddedNoise = options.getOrElse("dataAddedNoise","-1").toDouble
    solverOptions.dataNoiseOnlyTest = options.getOrElse("dataNoiseOnlyTest","false").toBoolean
    solverOptions.dataGenTrainSize = options.getOrElse("dataGenTrainSize","40").toInt
    solverOptions.dataGenTestSize = options.getOrElse("dataGenTestSize",solverOptions.dataGenTrainSize.toString).toInt
    solverOptions.dataRandSeed = options.getOrElse("dataRandSeed","-1").toInt
    solverOptions.dataGenCanvasSize = options.getOrElse("dataGenCanvasSize","16").toInt
    solverOptions.numClasses = options.getOrElse("numClasses","24").toInt //TODO this only makes sense if you end up with the MSRC dataset 
    if(solverOptions.dataGenSparsity> 0)
      solverOptions.dataWasGenerated=true
    
    
    
    /**
     * Some local overrides
     */
    if (runLocally) {
     // solverOptions.sampleFrac = 0.2
      solverOptions.enableOracleCache = false
      solverOptions.oracleCacheSize = 100
      solverOptions.stoppingCriterion = RoundLimitCriterion
      //solverOptions.roundLimit = 2
      solverOptions.enableManualPartitionSize = true
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false
      
      //solverOptions.debug = true
      //solverOptions.debugMultiplier = 1
    }
    
    // (Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]], Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]]) 

    
    def getMSRC():(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]])={
          val (oldtrainData, oldtestData) = ImageSegmentationUtils.loadMSRC("../data/generated/MSRC_ObjCategImageDatabase_v2")
      
        val graphTrainD = for (i <- 0 until oldtrainData.size) yield {
          val (gTrain, gLabel) = GraphUtils.convertOT_msrc_toGraph(oldtrainData(i).pattern, oldtrainData(i).label, solverOptions.numClasses)
          new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](gLabel, gTrain)
        }
        val graphTestD = for (i <- 0 until oldtestData.size) yield {
          val (gTrain, gLabel) = GraphUtils.convertOT_msrc_toGraph(oldtestData(i).pattern, oldtestData(i).label, solverOptions.numClasses)
          new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](gLabel, gTrain)
        }
    return (graphTrainD.toArray.toSeq, graphTestD.toArray.toSeq)
    }
    def genSquareNoiseD():(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]])= {
      val trainData = GraphUtils.genSquareBlobs(solverOptions.dataGenTrainSize,solverOptions.dataGenCanvasSize,solverOptions.dataGenSparsity,solverOptions.numClasses, if(solverOptions.dataNoiseOnlyTest) 0.0 else solverOptions.dataAddedNoise,solverOptions.dataRandSeed).toArray.toSeq
    val testData = GraphUtils.genSquareBlobs(solverOptions.dataGenTestSize,solverOptions.dataGenCanvasSize,solverOptions.dataGenSparsity,solverOptions.numClasses,solverOptions.dataAddedNoise,solverOptions.dataRandSeed).toArray.toSeq

     return (trainData,testData)
    }
    
    
   
    val (trainData,testData) = if(solverOptions.useMSRC) getMSRC() else genSquareNoiseD()
   
    
    if(solverOptions.useMSRC) solverOptions.numClasses = 24
    
    /*
    val trainData = GraphUtils.genSquareBlobs(solverOptions.dataGenTrainSize,solverOptions.dataGenCanvasSize,solverOptions.dataGenSparsity,solverOptions.numClasses, if(solverOptions.dataNoiseOnlyTest) 0.0 else solverOptions.dataAddedNoise,solverOptions.dataRandSeed).toArray.toSeq
    val testData = GraphUtils.genSquareBlobs(solverOptions.dataGenTestSize,solverOptions.dataGenCanvasSize,solverOptions.dataGenSparsity,solverOptions.numClasses,solverOptions.dataAddedNoise,solverOptions.dataRandSeed).toArray.toSeq
*/
    
    //TODO remove this debug (this could be made into a testcase )
    if(false){
        val first = trainData(0).pattern.getF(0).size
      for ( i <- 0 until trainData.size){
        for(s <- 0 until trainData(i).pattern.size){
          assert(trainData(i).pattern.getF(s).size==first)
        }
        
      }
      println("#Fun all is well")
    
    }
    /*
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
    */
      
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

        val myGraphSegObj = new GraphSegmentationClass(solverOptions.onlyUnary,MAX_DECODE_ITERATIONS,solverOptions.learningRate ,solverOptions.useMF,solverOptions.mfTemp,solverOptions.useNaiveUnaryMax)
    val trainer: StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] =
      new StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels](
        trainDataRDD,
        myGraphSegObj,
        solverOptions)

    val t0MTrain = System.currentTimeMillis()
    val model: StructSVMModel[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] = trainer.trainModel()
    val t1MTrain = System.currentTimeMillis()
    var avgTrainLoss = 0.0

    var count=0
    for (item <- trainData) {
      val prediction = model.predict(item.pattern)
      
      if(printImages){
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(item.label, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Train"+count+"trueRW_"+solverOptions.runName+".bmp")
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(prediction, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Train"+count+"predRW_"+solverOptions.runName+".bmp")
      }
      avgTrainLoss += myGraphSegObj.lossFn(item.label, prediction)
      count+=1
    }
    avgTrainLoss = avgTrainLoss / trainData.size
    println("\nTRAINING: Avg Loss : " + avgTrainLoss + " numItems " + trainData.size)
    //Test Error 
    var avgTestLoss = 0.0
    count=0
    for (item <- testData) {
      val prediction = model.predict(item.pattern)
      
      if(printImages){
            GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(item.label, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Test"+count+"trueRW_"+solverOptions.runName+".bmp")
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(prediction, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Test"+count+"predRW_"+solverOptions.runName+".bmp")
      }
      avgTestLoss += myGraphSegObj.lossFn(item.label, prediction)
      count+=1
    }
    avgTestLoss = avgTestLoss / testData.size
    println("\nTest Avg Loss : " + avgTestLoss + " numItems " + testData.size)
    
    println("#EndScore#,%d,%s,%s,%d,%.3f,%.3f,%s,%d,%d,%.3f,%s,%d,%d,%s,%s,%d,%s,%f,%f,%d,%s".format(
        solverOptions.startTime, solverOptions.runName,solverOptions.gitVersion,(t1MTrain-t0MTrain),solverOptions.dataGenSparsity,solverOptions.dataAddedNoise,if(solverOptions.dataNoiseOnlyTest)"t"else"f",solverOptions.dataGenTrainSize,solverOptions.dataGenCanvasSize,solverOptions.learningRate,if(solverOptions.useMF)"t"else"f",solverOptions.numClasses,MAX_DECODE_ITERATIONS,if(solverOptions.onlyUnary)"t"else"f",if(solverOptions.debug)"t"else"f",solverOptions.roundLimit,if(solverOptions.dataWasGenerated)"t"else"f",avgTestLoss,avgTrainLoss,solverOptions.dataRandSeed ,solverOptions.useMSRC   ) )
        
  
    sc.stop()

  }

}