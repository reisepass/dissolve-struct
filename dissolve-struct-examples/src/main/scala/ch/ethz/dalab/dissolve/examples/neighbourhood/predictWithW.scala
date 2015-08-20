package ch.ethz.dalab.dissolve.examples.neighbourhood


import ch.ethz.dalab.scalaslic.SLIC
import ch.ethz.dalab.scalaslic.DatumCord
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
import ij._
import ij.io.Opener
import breeze.linalg._
import breeze.stats.DescriptiveStats._
import breeze.stats._
import breeze.numerics._
import breeze.linalg._
import breeze.util.JavaArrayOps
import ij.plugin.Duplicator
import scala.util.matching.Regex
import java.io.File
import scala.collection.mutable.HashMap
import java.util.concurrent.atomic.AtomicInteger
import scala.pickling.Defaults._
import scala.pickling.binary._
import scala.pickling.static._
import scala.io.Source
import java.io._
import scala.util.Random
import java.awt.image.ColorModel
import java.awt.Color
import scala.collection.mutable.ListBuffer
import ch.ethz.dalab.dissolve.examples.neighbourhood.startupUtils._
import ch.ethz.dalab.dissolve.optimization.SSGSolver


import ch.ethz.dalab.dissolve.classification.StructSVMModel


import ch.ethz.dalab.dissolve.classification.StructSVMModel

 

/**
 * @author mort
 */
object predictWithW {
  
  
  
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
    println("Options:: "+options)
    runStuff(options)
  }
  
  
  
  
    val featFn3 = (image:ImageStack,mask:Array[Array[Array[Int]]],numSupPix:Int,sO:SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels])=>{
     val xDim = mask.length
     val yDim = mask(0).length
     val zDim = mask(0)(0).length
     
    
   val histBinsPerCol = sO.featHistSize/3
   val histBinsPerGray = sO.featHistSize
   val histCoGrayBins = sO.featCoOcurNumBins
   val histCoColorBins = sO.featCoOcurNumBins/3
   
 
   val out = Array.fill(numSupPix){ new  ListBuffer[Double]}
      val bitDep = image.getBitDepth()
        val isColor = if(bitDep==8) false else true //TODO maybe this is not the best way to check for color in the image
        //TODO the bit depth should give me the max value which the hist should span over 
        
        if(isColor){
          
          
          
          if(sO.featIncludeMeanIntensity){
          colorAverageIntensity1(image,mask).foreach( (a:(Int,Double))=> {
            out(a._1)++=List(a._2) 
            
          })
          }
          
          if(sO.featHistSize>0){
            assert(sO.featHistSize%3==0)
             colorhist(image,mask,histBinsPerCol,255 / histBinsPerCol).foreach( (a:(Int,Array[Double]))=> {out(a._1)++=a._2 })
          }
         if((sO.featCoOcurNumBins>0)){
            coOccurancePerSuperRGB(mask, image, numSupPix, 2).foreach( (a:(Int,Array[Double]))=> {out(a._1)++=a._2 })
          }
                 
          if(sO.featAddIntensityVariance){
           colorIntensityVariance(image,mask,numSupPix).foreach( (a:(Int,Double))=> {out(a._1)++=List(a._2) })
         }
         if(sO.featAddOffsetColumn){
          ((0 until numSupPix) zip List.fill(numSupPix){1.0}).toMap.foreach( (a:(Int,Double))=> {out(a._1)++=List(a._2) })
         }
          
          
        }
       else{
         
         if(sO.featIncludeMeanIntensity){
          greyAverageIntensity1(image,mask).foreach( (a:(Int,Double))=> {
            
            out(a._1)++=List(a._2) })
          }
          
          if(sO.featHistSize>0&&(!sO.featUseStdHist)){
            greyHist(image,mask,histBinsPerGray,255 / (histBinsPerGray)).foreach( (a:(Int,Array[Double]))=> {out(a._1)++=a._2 })
          }
         if((sO.featCoOcurNumBins>0)){
            greyCoOccurancePerSuper(image, mask, histCoGrayBins).foreach( (a:(Int,Array[Double]))=> {out(a._1)++=a._2 })
          }
                 
          if(sO.featAddIntensityVariance){
           greyIntensityVariance(image,mask,numSupPix).foreach( (a:(Int,Double))=> {out(a._1)++=List(a._2) })
         }
         if(sO.featAddOffsetColumn){
          ((0 until numSupPix) zip List.fill(numSupPix){1.0}).toMap.foreach( (a:(Int,Double))=> {out(a._1)++=List(a._2) })
         }
         
        
          
       }   
      
      
     val oo = (for ( i<- 0 until numSupPix) yield{ out(i).toArray}).toArray
    
    if(sO.standardizeFeaturesByColumn){
     val oodMat = JavaArrayOps.array2DToDm(oo)
     for(i<- 0 until oo(0).length){
       val stdCol=standardize(oodMat(::,i))
       oodMat(::,i):=stdCol
     }
      val outAgain=JavaArrayOps.dmDToArray2(oodMat)
      outAgain
    }
    else{
      oo
    }
   }
    
     def standardize (a:DenseVector[Double]):DenseVector[Double]={
       val mysum = sum(a)
       val mean = mysum/a.length
       val mySD = stddev(a)
       if(mySD==0.0 )
         a:-=mean
       else
       (a:-=mean):*=(1/mySD)
     }
  val afterFeatFn1 =(image:ImageStack,mask:Array[Array[Array[Int]]], nodes: IndexedSeq[Node[Vector[Double]]] , numSupPix:Int,sO:SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels])=>{
    
    
 
    if(nodes(0).avgValue==Double.MinValue){
        if(sO.isColor){
          val intens = colorAverageIntensity1(image, mask, sO.maxColorValue)
          for( i <- 0 until numSupPix)
            nodes(i).avgValue=intens.get(i).get
        }
        else{
          val intens = greyAverageIntensity1(image, mask, sO.maxColorValue)
          for( i <- 0 until numSupPix)
            nodes(i).avgValue=intens.get(i).get
        }
        
      }
    
    val greyStdHist = if(sO.featUseStdHist) greyHistvStd(image, mask, numSupPix, sO) else Array.fill(numSupPix){Array[Double]()}
    
    
    val neighHists = if(sO.featNeighHist){
      assert(sO.featHistSize>0,"You selected the option to add feature neighbour histograms but did not specify a histogram size")
      
      
      val perCHistSize = if(sO.isColor) sO.featHistSize/3 else sO.featHistSize
      val binWidth = sO.maxColorValue/perCHistSize
      val neighHists= if(sO.isColor){
        val histPerSup=colorhist(image,mask,sO.featHistSize/3,binWidth,sO.maxColorValue)

        assert(histPerSup.get(0).get.length==sO.featHistSize)
        val neighHistPerSuper = for( i <- 0 until numSupPix) yield {
          val neigh=nodes(i).connections
          var sumHists = DenseVector.zeros[Double](sO.featHistSize)
          
          neigh.foreach { neighID => {
            sumHists=sumHists+DenseVector(histPerSup.get(neighID).get)
          }}
          sumHists:*=(1/(neigh.size.asInstanceOf[Double]))
          normalize(sumHists).toArray
          }
        
         if(sO.standardizeFeaturesByColumn){
        val oodMat = JavaArrayOps.array2DToDm(neighHistPerSuper.toArray)
         for(i<- 0 until sO.featHistSize){
           val stdCol=standardize(oodMat(::,i))
           oodMat(::,i):=stdCol
         }
          val outAgain=JavaArrayOps.dmDToArray2(oodMat)
          outAgain
         }
         else{
           neighHistPerSuper.toArray
         }
      }
      
      
      else{
        val histPerSup=  if(sO.featUseStdHist) greyStdHist else greyHistv2(image,mask,numSupPix,sO)
        
        val neighHistPerSuper = for( i <- 0 until numSupPix) yield {
          val neigh=nodes(i).connections
          val sumHists = DenseVector.zeros[Double](histPerSup(0).length)
          
          neigh.foreach { neighID => {
            sumHists+=DenseVector(histPerSup(neighID))
          }}
          sumHists:*=(1/(neigh.size.asInstanceOf[Double]))
          normalize(sumHists).toArray
          }
        if(sO.standardizeFeaturesByColumn){
        val oodMat = JavaArrayOps.array2DToDm(neighHistPerSuper.toArray)
     for(i<- 0 until sO.featHistSize){
       val stdCol=standardize(oodMat(::,i))
       oodMat(::,i):=stdCol
     }
      val outAgain=JavaArrayOps.dmDToArray2(oodMat)
      outAgain
      }
        else{
          neighHistPerSuper.toArray
        }
      }
      
      neighHists
    }
    else{
        Array.fill(numSupPix){Array[Double]()}
      }
    
    val superSizes= if(sO.featAddSupSize){
       val tmpSizes= countSuperPixelSizes(mask,numSupPix)
       for ( i <- 0 until numSupPix)
         nodes(i).size=tmpSizes(i)
         val ttt = tmpSizes.map(x => x.toDouble)
      standardize(DenseVector(ttt ))
    }
    else { 
      DenseVector(Array.fill(numSupPix){(-1.0)})
    }
    
    if(sO.featUniqueIntensity){
      
      
      
      val gintens = new ListBuffer[Double]
      val gneighAvg = new ListBuffer[Double]
      val gneighVar = new ListBuffer[Double]
      val gneigh2hAvg = new ListBuffer[Double]
      val gneighh2Var = new ListBuffer[Double]
      
      for( i <- 0 until numSupPix){
        val xi = nodes(i)
        gintens++=List(xi.avgValue)
        val neighIntens = xi.connections.map { neighId => nodes(neighId).avgValue }.toList
        val neighAvg=sum(neighIntens)/neighIntens.length 
        nodes(i).neighMean= neighAvg
        gneighAvg++=List(neighAvg)
        
        val allIntens = List(neighIntens, List(xi.avgValue)).flatten
        val neighVar = variance(allIntens)
        nodes(i).neighVariance = neighVar
        gneighVar++=List(neighVar)
        
        if(sO.featUnique2Hop){
        val hop2Neigh = xi.connections.map{ neighId => nodes(neighId).connections.toList}.toList.flatten.toSet
        val hop2Intens = hop2Neigh.map { neighId => nodes(neighId).avgValue }.toList
        val hop2Avg = sum(hop2Intens)/hop2Intens.length
        nodes(i).hop2NeighMean=hop2Avg
        gneigh2hAvg++=List(hop2Avg)
        val hop2Var = variance(hop2Intens)
        nodes(i).hop2NeighVar=hop2Var
        gneighh2Var++=List(hop2Var)
        }
        else{
          gneigh2hAvg++=List(0.0)
          gneighh2Var++=List(0.0)
        }
        
      }
      
     

      val nor_intens =  standardize(DenseVector(gintens.toArray))
      val nor_neighAvg = standardize(DenseVector(gneighAvg.toArray))
      val nor_neighVar = standardize(DenseVector(gneighVar.toArray))
      val nor_neigh2hAvg = standardize(DenseVector(gneigh2hAvg.toArray))
      val nor_neighh2Var = standardize(DenseVector(gneighh2Var.toArray))
      
      val nor_iDif = standardize(nor_intens-nor_neighAvg)
      val nor_iRatio = standardize(nor_intens:/nor_neighAvg)
      val nor_iNDif = standardize(nor_intens-nor_neigh2hAvg)
      val nor_iNRatio = standardize(nor_intens:/nor_neigh2hAvg)
      val nor_iNN2Dif = standardize(nor_neigh2hAvg-nor_intens)
      val nor_iNN2Ratio = standardize(nor_neigh2hAvg/nor_intens)
      val nor_iNN2VarDif = standardize(nor_neighh2Var-nor_neighVar)
      val nor_iNN2VarRatio = standardize(nor_neighh2Var/nor_neighVar)
      
      
      val out = nodes.map { old =>  { 
        
        val i = old.idx
        val f = Vector(Array(nor_iDif(old.idx),nor_iRatio(old.idx),nor_neighVar(old.idx))++
            (if(sO.featUnique2Hop) Array(nor_iNDif(old.idx),nor_iNRatio(old.idx),nor_neighh2Var(old.idx),
                (nor_iNN2Dif(old.idx)),nor_iNN2Ratio(old.idx),
                (nor_neighh2Var(old.idx)-nor_neighVar(old.idx)),nor_iNN2VarRatio(old.idx)) else Array[Double]() ) 
            ++old.features.toArray++neighHists(old.idx)++greyStdHist(old.idx)++(if(sO.featAddSupSize)Array(superSizes(i).toDouble) else Array[Double]())  )
        new Node[Vector[Double]](i, f,old.connections,nor_intens(i),nor_neighAvg(i),nor_neighVar(i),nor_neigh2hAvg(i),nor_neighh2Var(i),old.size)
      }}
      
      out.toArray
    }
    else
      nodes.toArray
  } 
  
      
  type xData = GraphStruct[Vector[Double], (Int, Int, Int)]
  type yLabels = GraphLabels
  
  
  def featureFn(xDat: xData, yDat: yLabels): Vector[Double] = {
    assert(xDat.graphNodes.size == yDat.d.size)
    

    val xFeatures = xDat.getF(0).size
    val numClasses = yDat.numClasses

    val unaryFeatureSize = xFeatures * numClasses
    
    val phi =  DenseVector.zeros[Double](unaryFeatureSize)

    
    val unary = DenseVector.zeros[Double](unaryFeatureSize)
    for (idx <- 0 until yDat.d.size) {
      val label = yDat.d(idx)
      val startIdx = label * xFeatures
      val endIdx = startIdx + xFeatures
      val curF =xDat.getF(idx) 
      unary(startIdx until endIdx) :=curF + unary(startIdx until endIdx)
    }

    
      phi(0 until (unaryFeatureSize)) := unary
    

    

   phi
  }
  
  
  
  def runStuff(options: Map[String, String]) {
   
    
    printMemory()

    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")
    val runLocally: Boolean = options.getOrElse("local", "false").toBoolean
    val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    var gitV = "noGitPresent" 
    try { gitV=("git rev-parse HEAD"!!).replaceAll("""(?m)\s+$""", "")}catch {
      case t: Throwable => t.printStackTrace() // TODO: handle error
    }
    
    val experimentName:String = options.getOrElse("runName", "UnNamed")
    
    val msrcDir: String = "../data/generated"

    val appName: String = "ImageSegGraph"

    val printImages: Boolean = options.getOrElse("printImages","false").toBoolean
    
    val sO: SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] = new SolverOptions()
    sO.gitVersion = gitV
    sO.runName = experimentName
    sO.startTime = System.currentTimeMillis
    sO.debugMultiplier=options.getOrElse("debugMultiplier","1").toInt
    sO.roundLimit = options.getOrElse("roundLimit", "5").toInt // After these many passes, each slice of the RDD returns a trained model
    sO.debug = options.getOrElse("debug", "false").toBoolean
    sO.lambda = options.getOrElse("lambda", "0.01").toDouble
    sO.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    sO.doLineSearch = options.getOrElse("linesearch", "true").toBoolean
    sO.debug = options.getOrElse("debug", "false").toBoolean
    sO.debugWeightUpdate=options.getOrElse("debugWeightUpdate","false").toBoolean
    sO.onlyUnary = options.getOrElse("onlyUnary", "false").toBoolean
    val MAX_DECODE_ITERATIONS:Int = options.getOrElse("maxDecodeItr",  (if(sO.onlyUnary) 100 else 1000 ).toString ).toInt
    val MAX_DECODE_ITERATIONS_MF_ALT:Int = options.getOrElse("maxDecodeItrMF",  (MAX_DECODE_ITERATIONS).toString ).toInt
    sO.sample = options.getOrElse("sample", "frac")
    sO.sampleFrac = options.getOrElse("samplefrac", "1").toDouble
    sO.dbcfwSeed = options.getOrElse("dbcfwSeed","-1").toInt
    sO.randSeed = options.getOrElse("oldRandSeed","42").toInt
    sO.sampleWithReplacement = options.getOrElse("samplewithreplacement", "false").toBoolean
    sO.enableManualPartitionSize = options.getOrElse("manualrddpart", "false").toBoolean
    sO.NUM_PART = options.getOrElse("numpart", "2").toInt
    sO.enableOracleCache = options.getOrElse("enableoracle", "false").toBoolean
    sO.oracleCacheSize = options.getOrElse("oraclesize", "5").toInt
    sO.useClassFreqWeighting =  options.getOrElse("weightedClassFreq","false").toBoolean
    sO.useMF=options.getOrElse("useMF","false").toBoolean
    sO.useNaiveUnaryMax = options.getOrElse("useNaiveUnaryMax","false").toBoolean
    sO.learningRate = options.getOrElse("learningRate","0.1").toDouble
    sO.mfTemp=options.getOrElse("mfTemp","5.0").toDouble
    sO.debugInfoPath = options.getOrElse("debugpath", debugDir + "/imageseg-%d.csv".format(System.currentTimeMillis()))
    sO.useMSRC = options.getOrElse("useMSRC","false").toBoolean
    sO.dataGenSparsity = options.getOrElse("dataGenSparsity","-1").toDouble
    sO.dataAddedNoise = options.getOrElse("dataAddedNoise","0.0").toDouble
    sO.dataNoiseOnlyTest = options.getOrElse("dataNoiseOnlyTest","false").toBoolean
    sO.dataGenTrainSize = options.getOrElse("dataGenTrainSize","40").toInt
    sO.dataGenTestSize = options.getOrElse("dataGenTestSize",sO.dataGenTrainSize.toString).toInt
    sO.dataRandSeed = options.getOrElse("dataRandSeed","-1").toInt
    sO.dataGenCanvasSize = options.getOrElse("dataGenCanvasSize","16").toInt
    sO.numClasses = options.getOrElse("numClasses","24").toInt //TODO this only makes sense if you end up with the MSRC dataset 
    sO.generateMSRCSupPix = options.getOrElse("supMSRC","false").toBoolean
    sO.squareSLICoption = options.getOrElse("supSquareMSRC","false").toBoolean
    sO.trainTestEqual = options.getOrElse("trainTestEqual","false").toBoolean
    val GEN_NEW_SQUARE_DATA=options.getOrElse("GEN_NEW_SQUARE_DATA","false").toBoolean
    sO.isColor = options.getOrElse("isColor","true").toBoolean
    sO.superPixelSize = options.getOrElse("S","15").toInt
    sO.initWithEmpiricalTransProb = options.getOrElse("initTransProb","false").toBoolean
    sO.dataFilesDir = options.getOrElse("dataDir","###")
    assert(!sO.dataFilesDir.equals("###"),"Error,  Please specify a the 'dataDir' paramater")
    val tmpDir = sO.dataFilesDir.split("/")
    sO.dataSetName = tmpDir(tmpDir.length-1)
    sO.imageDataFilesDir = options.getOrElse("imageDir",  sO.dataFilesDir+"/Images")
    sO.groundTruthDataFilesDir = options.getOrElse("groundTruthDir",  sO.dataFilesDir+"/GroundTruth")
    sO.weighDownPairwise = options.getOrElse("weighDownPairwise","1.0").toDouble
    assert(sO.weighDownPairwise<=1.0 &&sO.weighDownPairwise>=0)
    sO.weighDownUnary = options.getOrElse("weighDownUnary","1.0").toDouble
    assert(sO.weighDownUnary<=1.0&&sO.weighDownUnary>=0)
    if(sO.squareSLICoption) sO.generateMSRCSupPix=true
    val DEBUG_COMPARE_MF_FACTORIE =  options.getOrElse("cmpEnergy","false").toBoolean  
    if(sO.dataGenSparsity> 0) sO.dataWasGenerated=true
    sO.LOSS_AUGMENTATION_OVERRIDE = options.getOrElse("LossAugOverride", "false").toBoolean
    sO.putLabelIntoFeat = options.getOrElse("labelInFeat","false").toBoolean
    sO.PAIRWISE_UPPER_TRI = options.getOrElse("PAIRWISE_UPPER_TRI","true").toBoolean
    sO.dataGenSquareNoise = options.getOrElse("dataGenSquareNoise","0.0").toDouble
    sO.dataGenSquareSize  = options.getOrElse("dataGenSquareSize","10").toInt
    sO.dataGenHowMany = options.getOrElse("dataGenHowMany","40").toInt
    sO.dataGenOsilNoise = options.getOrElse("dataGenOsilNoise","0.0").toDouble
    sO.dataGenGreyOnly = options.getOrElse("dataGenGreyOnly","false").toBoolean
    sO.compPerPixLoss = options.getOrElse("compPerPixLoss","false").toBoolean
    sO.dataGenEnforNeigh = options.getOrElse("dataGenEnforNeigh","false").toBoolean
    sO.dataGenNeighProb = options.getOrElse("dataGenNeighProb","1.0").toDouble  
    sO.slicCompactness = options.getOrElse("slicCompactness","-1.0").toDouble
    sO.featHistSize=options.getOrElse("featHistSize","9").toInt
    sO.slicNormalizePerClust=options.getOrElse("slicNormalizePerClust","true").toBoolean
    sO.featCoOcurNumBins=options.getOrElse("featCoOcurNumBins","6").toInt
    sO.debugPrintSuperPixImg = options.getOrElse("debugPrintSuperPixImg","false").toBoolean
    sO.useLoopyBP = options.getOrElse("useLoopyBP","false").toBoolean
    sO.useMPLP= options.getOrElse("useMPLP","false").toBoolean
    val GEN_TRY_TO_MATCH_GENERATED_DATA=options.getOrElse("GEN_TRY_TO_MATCH_GENERATED_DATA","false").toBoolean
    assert(sO.useMF||sO.useNaiveUnaryMax||sO.useMPLP||sO.useLoopyBP)
    implicit def bool2int(b:Boolean) = if (b) 1 else 0

    assert((sO.useMF:Int)+(sO.useNaiveUnaryMax:Int)+(sO.useMPLP:Int)+(sO.useLoopyBP:Int )==1)
    
    sO.inferenceMethod= if(sO.useMF) "MF" else if(sO.useNaiveUnaryMax) "NAIVE_MAX" else if(sO.useMPLP) "Factorie" else if(sO.useLoopyBP) "LoopyBP" else "NotFound"
    sO.featIncludeMeanIntensity = options.getOrElse("featMeanIntensity","false").toBoolean
    sO.modelPairwiseDataDependent = options.getOrElse("modelPairwiseDataDependent", "false").toBoolean
    if(sO.modelPairwiseDataDependent){
      assert(sO.useLoopyBP,"if you want to use modelPairwiseDataDependent you have to set useLoopyBP=true")
   //   assert(sO.featIncludeMeanIntensity)
    }
    if(sO.slicCompactness==(-1.0)){
      sO.slicCompactness=sO.superPixelSize.asInstanceOf[Double]
    }
    sO.featAddOffsetColumn=options.getOrElse("featAddOffsetColumn","false").toBoolean
    sO.featAddIntensityVariance=options.getOrElse("featAddIntensityVariance","false").toBoolean
    sO.recompFeat=options.getOrElse("recompFeat","false").toBoolean
    sO.featUnique2Hop=options.getOrElse("featUnique2Hop","false").toBoolean
    sO.featUniqueIntensity = options.getOrElse("featUniqueIntensity","false").toBoolean
    
    sO.dataDepUseIntensity=options.getOrElse("dataDepUseIntensity","false").toBoolean
    if(sO.dataDepUseIntensity)
      sO.dataDepMeth="dataDepUseIntensity"
    sO.dataDepUseIntensityByNeighSD=options.getOrElse("dataDepUseIntensityByNeighSD","false").toBoolean
    if(sO.dataDepUseIntensityByNeighSD)
      sO.dataDepMeth="dataDepUseIntensityByNeighSD"
    if(sO.dataDepUseIntensityByNeighSD)
      assert(sO.featAddIntensityVariance)
    sO.dataDepUseIntensityBy2NeighSD=options.getOrElse("dataDepUseIntensityBy2NeighSD","false").toBoolean
    if(sO.dataDepUseIntensityBy2NeighSD)
      sO.dataDepMeth="dataDepUseIntensityBy2NeighSD"
    if(sO.dataDepUseIntensityBy2NeighSD)
      assert(sO.featAddIntensityVariance)
    sO.dataDepUseUniqueness=options.getOrElse("dataDepUseUniqueness","false").toBoolean
    if(sO.dataDepUseUniqueness)
      sO.dataDepMeth="dataDepUseUniqueness"
    if(sO.dataDepUseUniquenessInOtherNeighbourhood)
      sO.dataDepMeth="dataDepUseUniquenessInOtherNeighbourhood"
    assert((sO.dataDepUseUniqueness:Int)+(sO.dataDepUseIntensityBy2NeighSD:Int)+(sO.dataDepUseIntensityByNeighSD:Int)+(sO.dataDepUseIntensity:Int )+(sO.dataDepUseUniquenessInOtherNeighbourhood:Int )<=1)
   

    sO.slicMinBlobSize = options.getOrElse("slicMinBlobSize","-1").toInt
    sO.standardizeFeaturesByColumn=options.getOrElse("standardizeFeaturesByColumn","false").toBoolean
    sO.featNeighHist = options.getOrElse("featNeighHist","false").toBoolean
    sO.featUseStdHist = options.getOrElse("featUseStdHist","false").toBoolean
    sO.preStandardizeImagesFirst = options.getOrElse("preStandardizeImagesFirst" , "false").toBoolean
    sO.numDataDepGraidBins= options.getOrElse("numDataDepGraidBins","5").toInt
    sO.loopyBPmaxIter = options.getOrElse("loopyBPmaxIter","10").toInt
    sO.alsoWeighLossAugByFreq = options.getOrElse("alsoWeighLossAugByFreq","false").toBoolean
    val SPLIT_IMAGES = options.getOrElse("SPLIT_IMAGES","false").toBoolean
    sO.splitImagesBy = options.getOrElse("splitImagesBy","-1").toInt
    sO.optimizeWithSubGraid = options.getOrElse("optimizeWithSubGraid","false").toBoolean
    sO.featAddSupSize = options.getOrElse("featAddSupSize","false").toBoolean
    if(sO.featAddSupSize)
      assert(sO.featUniqueIntensity,"If you set featAddSupSize to true you have to also set featUniqueIntensity to true")
    sO.pairwiseModelPruneSomeEdges= options.getOrElse("pairwiseModelPruneSomeEdges","0.0").toDouble
      
    //
    //
    //Actual W params 
    //
    //  
    
    val useRandomW=options.getOrElse("useRandomW","true").toBoolean
    val useZeroW = options.getOrElse("useZeroW","false").toBoolean
      
      
    val prefix = if(useRandomW) "RandomW_" else "ZeroW"
    sO.inferenceMethod=prefix+sO.inferenceMethod
    /**
     * Some local overrides
     */
    if (runLocally) {
     // solverOptions.sampleFrac = 0.2
      sO.enableOracleCache = false
      sO.oracleCacheSize = 100
      sO.stoppingCriterion = RoundLimitCriterion
      //solverOptions.roundLimit = 2
      sO.enableManualPartitionSize = true
      sO.NUM_PART = 1
      sO.doWeightedAveraging = false
      
      //solverOptions.debug = true
      //solverOptions.debugMultiplier = 1
    }
    
   
   if(SPLIT_IMAGES){
     assert(!GEN_NEW_SQUARE_DATA,"We hault the execution here because it seems unlikely that one would set GEN_NEW_SQUARE_DATA and SPLIT_IMAGES to true, please adjust the input arguments")
     assert(!GEN_TRY_TO_MATCH_GENERATED_DATA,"We hault the execution here because it seems unlikely that one would set GEN_TRY_TO_MATCH_GENERATED_DATA and SPLIT_IMAGES to true, please adjust the input arguments")
     if(sO.splitImagesBy==(-1))
       sO.splitImagesBy=2
     splitLargeTiffStack(sO.dataFilesDir, sO.splitImagesBy)
   }
    
   if(GEN_NEW_SQUARE_DATA||GEN_TRY_TO_MATCH_GENERATED_DATA){
           if(sO.dataGenSparsity==(-1))
             sO.dataGenSparsity=1.0/sO.numClasses
           val lotsofBMP = Option(new File(sO.dataFilesDir+"/Images").list).map(_.filter(_.endsWith(".bmp")))
           if(GEN_NEW_SQUARE_DATA)
             assert(lotsofBMP.isEmpty,"You tried to generate data into a folder which already has some in it")
           if(GEN_TRY_TO_MATCH_GENERATED_DATA){
             
           
            
             val dirName="generatedData__"+sO.runName+"_"+(if(sO.dataGenGreyOnly) "grey" else "color")+"_"+sO.dataGenHowMany+"_"+sO.dataGenCanvasSize+"_"+sO.dataGenSparsity+"_"+sO.numClasses+"_"+sO.dataGenSquareNoise+"_"+sO.dataAddedNoise+"_"+sO.dataGenOsilNoise+"_"+sO.dataGenNeighProb+"_"+sO.superPixelSize+"_"+sO.dataRandSeed+"___";
             val tmpDir = sO.dataFilesDir.split("/")
              sO.dataSetName = dirName
              tmpDir(tmpDir.length-1)=dirName
               //sO.dataFilesDir=tmpDir.mkString("/")
              val worlD =  new File(".").getCanonicalPath()
               sO.dataFilesDir=worlD+"/"+dirName
             
              
    sO.imageDataFilesDir = options.getOrElse("imageDir",  sO.dataFilesDir+"/Images")
    sO.groundTruthDataFilesDir = options.getOrElse("groundTruthDir",  sO.dataFilesDir+"/GroundTruth")
  
           }
            val lotsofBMP2= Option(new File(sO.dataFilesDir+"/Images").list).map(_.filter(_.endsWith(".bmp")))
            
        if(lotsofBMP2.isEmpty){
        if(sO.dataGenGreyOnly)
          GraphUtils.genGreyfullSquaresDataSuperNoise(sO.dataGenHowMany,sO.dataGenCanvasSize,sO.dataGenSquareSize,sO.dataGenSparsity,sO.numClasses,
            sO.dataGenSquareNoise,sO.dataAddedNoise,
            sO.dataFilesDir,sO.dataRandSeed,
            sO.dataGenOsilNoise,sO.superPixelSize)
        else{
          if(!sO.dataGenEnforNeigh){
            GraphUtils.genColorfullSquaresDataSuperNoise(sO.dataGenHowMany,sO.dataGenCanvasSize,sO.dataGenSquareSize,sO.dataGenSparsity,sO.numClasses,
            sO.dataGenSquareNoise,sO.dataAddedNoise,
            sO.dataFilesDir,sO.dataRandSeed,
            sO.dataGenOsilNoise,sO.superPixelSize)
          }
          else{
            val neighRules = Array.fill(sO.numClasses,sO.numClasses){true}
            for( x<- 0 until sO.numClasses ;y<-0 until sO.numClasses){
              //if(x!=y)
              if(x!=0&&y!=0)
                if(Random.nextDouble()>sO.dataGenNeighProb){
                  neighRules(x)(y)=false
                  neighRules(y)(x)=false
                }
              
            }
            GraphUtils.genColorfullSquaresDataSuperNoiseEnforceNeigh(neighRules,sO.dataGenHowMany,sO.dataGenCanvasSize,sO.dataGenSquareSize,sO.dataGenSparsity,sO.numClasses,
            sO.dataGenSquareNoise,sO.dataAddedNoise,
            sO.dataFilesDir,sO.dataRandSeed,
            sO.dataGenOsilNoise,sO.superPixelSize)
          }
        }
        }
        else{
          println("##WARNING## No new data was generated because we expect that the synthetic data generator would have deterministically had the same output DIR:: "+sO.dataFilesDir)
        }
    }
    
    val runCFG=new File(sO.dataFilesDir+"/"+sO.runName+"_run.cfg")
    if(!runCFG.exists()){
      val pw = new PrintWriter(new File(sO.dataFilesDir+"/"+sO.runName+"_run.cfg"))
      pw.write(options.toString)
      pw.write("\n")
      pw.write(sO.toString())
      pw.close
    }
    else{
      val pw = new PrintWriter(new File(sO.dataFilesDir+"/"+sO.runName+"_"+System.currentTimeMillis()+"_run.cfg"))
      pw.write(options.toString)
      pw.write("\n")
      pw.write(sO.toString())
      pw.close
    }
       
    
  
   val (trainData,testData, colorlabelMap, classFreqFound,transProb, newSo) = genGraphFromImages(sO,featFn3,afterFeatFn1)
    
    
   println("Train Size:"+trainData.size)
    println("Test Size:"+testData.size)
    
     if (runLocally) {
       sO.testData=Some(testData)
     }

    println("----- First Example of each Class -----")
    for( lab <- 0 until sO.numClasses){
      
      var labFound = -1; 
      
        val pat = trainData(0).pattern
        val labData = trainData(0).label
        val name = trainData(0).pattern.originMapFile
        var xi = 0
        while( labFound<= 10 && xi<labData.d.size){
          if(labData.d(xi)==lab){
            labFound +=1
            println("F of l_"+lab+"\t<"+pat.getF(xi)+">")
          }
          xi+=1
        }
        if(labFound==(-1)){
          println("F of l_"+lab+"\tNOT FOUND in first image ")
        }
      
    }
    
    
    
     if(sO.initWithEmpiricalTransProb){
      val featureSize =trainData(0).pattern.getF(0).size
      val unaryWeight =Array.fill(sO.numClasses*featureSize){0.0}
      val pairwiseWeight =  (new DenseVector(transProb.flatten)):/(10.0)
      val initWeight = unaryWeight++pairwiseWeight.toArray
      sO.initWeight=initWeight
     }
      
    
    
    /*
    for( i <- 0 until 10){
      val pat = trainData(i).pattern
      val primMat = GraphUtils.flatten3rdDim(GraphUtils.reConstruct3dMat(  trainData(i).label, pat.dataGraphLink,solverOptions.dataGenCanvasSize,solverOptions.dataGenCanvasSize,1))
      println(primMat.deep.mkString("\n"))
      println("------------------------------------------------")
    }
    * 
    */
    
   
    
    val myGraphSegObj= new GraphSegmentationClass(sO.onlyUnary,MAX_DECODE_ITERATIONS,
            sO.learningRate ,sO.useMF,sO.mfTemp,sO.useNaiveUnaryMax,
            DEBUG_COMPARE_MF_FACTORIE,MAX_DECODE_ITERATIONS_MF_ALT,sO.runName,
            if(sO.useClassFreqWeighting) classFreqFound else null,
            sO.weighDownUnary,sO.weighDownPairwise, sO.LOSS_AUGMENTATION_OVERRIDE,
            false,sO.PAIRWISE_UPPER_TRI,sO.useMPLP,sO.useLoopyBP,sO.loopyBPmaxIter,sO.alsoWeighLossAugByFreq,sO) 

    

    println(sO.toString())
    val samplePoint = trainData(0)
    
    val rand2 = if(sO.randSeed!=(-1)) new Random(sO.randSeed) else new Random()
    
    val d: Int = myGraphSegObj.featureFn(samplePoint.pattern, samplePoint.label).size
    
    val weights = if(useRandomW) Vector((0 until d).map( a=>rand2.nextDouble).toArray) else Vector(Array.fill(d){0.0})
    
     var model: StructSVMModel[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] = new StructSVMModel[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] (  weights, 0.0,
      DenseVector.zeros[Double](d), myGraphSegObj, sO.numClasses)
      

    var avgTrainLoss = 0.0
    var avgPerPixTrainLoss = 0.0
    var count=0
    val invColorMap = colorlabelMap.map(_.swap)
    
    for (item <- trainData) {
      val prediction = model.predict(item.pattern)
      
      
      if(printImages){
        
        /*
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(item.label, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Train"+count+"trueRW_"+solverOptions.runName+".bmp")
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(prediction, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Train"+count+"predRW_"+solverOptions.runName+".bmp")
      */
        val tmpPath = item.pattern.originMapFile
                val pathArray = item.pattern.originMapFile.split("/")
                val lastpart = pathArray(pathArray.length-1)
        val fileNameExt = lastpart.split("\\.")
        val fileName = fileNameExt(0)
        
        val tPrint = System.currentTimeMillis()
        GraphUtils.printBMPFromGraphInt(item.pattern,prediction,0,fileName+sO.runName+"_"+count+"-train_predict",colorMap=invColorMap)
        GraphUtils.printBMPFromGraphInt(item.pattern,item.label,0,fileName+sO.runName+"_"+count+"-train_true",colorMap=invColorMap)
       
        //val supPixToOrig = GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](item.pattern.originMapFile)
        //val origTrueLabel = GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](item.label.originalLabelFile)
    
      }
     
     avgTrainLoss += myGraphSegObj.lossFn(item.label, prediction) //TODO change this so that it tests the original pixel labels 
     val tFindLoss = System.currentTimeMillis()
    if(sO.compPerPixLoss)
       avgPerPixTrainLoss +=  GraphUtils.lossPerPixelInt(item.pattern.originMapFile, item.label.originalLabelFile,prediction,colorMap=colorlabelMap)
   //  println("PerPixel loss counting"+(System.currentTimeMillis()-tFindLoss))
      count+=1
    }
    avgTrainLoss = avgTrainLoss / trainData.size
    avgPerPixTrainLoss = avgPerPixTrainLoss/trainData.size
    println("\nTRAINING: Avg Loss : " + avgTrainLoss +") PerPix("+avgPerPixTrainLoss+ ") numItems " + trainData.size)
    //Test Error 
    var avgTestLoss = 0.0
    var avgPerPixTestLoss = 0.0
    count = 0
    
    def lossPerLabel(correct:GraphLabels,predict:GraphLabels):( Vector[Double],Vector[Double],Vector[Double])={
      val labelOccurance = Vector(Array.fill(sO.numClasses){0.0})
    val labelIncorrectlyUsed = Vector(Array.fill(sO.numClasses){0.0})
    val labelNotRecog = Vector(Array.fill(sO.numClasses){0.0})
     for ( i <- 0 until correct.d.size){
       val cLab = correct.d(i)
       val pLab = predict.d(i)
       labelOccurance(cLab)+=1.0
       if(cLab != pLab){
         labelNotRecog(cLab)+=1.0
         labelIncorrectlyUsed(pLab)+=1.0
       }
     }
     (labelOccurance,labelIncorrectlyUsed,labelNotRecog)
    }
    val labelOccurance = Vector(Array.fill(sO.numClasses){0.0})
    val labelIncorrectlyUsed = Vector(Array.fill(sO.numClasses){0.0})
    val labelNotRecog = Vector(Array.fill(sO.numClasses){0.0})
    
    
    for (item <- testData) {
      val prediction = model.predict(item.pattern)
      
      if(printImages){
        /*
            GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(item.label, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Test"+count+"trueRW_"+solverOptions.runName+".bmp")
      GraphUtils.printBMPfrom3dMat(GraphUtils.flatten3rdDim( GraphUtils.reConstruct3dMat(prediction, item.pattern.dataGraphLink,
      item.pattern.maxCoord._1+1, item.pattern.maxCoord._2+1, item.pattern.maxCoord._3+1)),"Test"+count+"predRW_"+solverOptions.runName+".bmp")
      
      * 
      */
        /*
        val tmpPath = item.pattern.originMapFile
                val pathArray = item.pattern.originMapFile.split("/")
                val lastpart = pathArray(pathArray.length-1)
        val fileNameExt = lastpart.split("\\.")
        val fileName = fileNameExt(0)
        GraphUtils.printBMPFromGraph(item.pattern,prediction,0,fileName+solverOptions.runName+"_"+count+"_test_predict",colorMap=invColorMap)
       GraphUtils.printBMPFromGraph(item.pattern,item.label,0,fileName+solverOptions.runName+"_"+count+"_test_true",colorMap=invColorMap)
        */
      }
  
      if(sO.compPerPixLoss)
      avgPerPixTestLoss +=  GraphUtils.lossPerPixelInt(item.pattern.originMapFile, item.label.originalLabelFile,prediction,colorMap=colorlabelMap)
      avgTestLoss += myGraphSegObj.lossFn(item.label, prediction)
      val perLabel = lossPerLabel(item.label, prediction)
      labelOccurance+=perLabel._1
      labelIncorrectlyUsed+=perLabel._2
      labelNotRecog+=perLabel._3
      count += 1
    }
    avgTestLoss = avgTestLoss / testData.size
    avgPerPixTestLoss = avgPerPixTestLoss/ testData.size
    val vocScores = (0 until labelOccurance.length).toList.map( idx => {
      labelOccurance(idx)/(labelOccurance(idx)+labelNotRecog(idx)+labelIncorrectlyUsed(idx))
    }).toArray
    
    
    println("\nTest Avg Loss : Sup(" + avgTestLoss +") PerPix("+avgPerPixTestLoss+ ") numItems " + testData.size)
    println()
    println("Occurances:           \t"+labelOccurance)
    println("labelIncorrectlyUsed: \t"+labelIncorrectlyUsed)
    println("labelNotRecog:        \t"+labelNotRecog)
    println("Label to Color Mapping "+colorlabelMap)
    println("Internal Class Freq:  \t"+classFreqFound)
    
    
     def bToS( a:Boolean)={if(a)"t"else"f"}
       
    
    val newLabels = ","+sO.sampleFrac+","+ (if(sO.doWeightedAveraging) "t" else "f")+","+ 
            (if(sO.onlyUnary) "t" else "f") +","+(if(sO.squareSLICoption) "t" else "f")+","+ sO.superPixelSize+","+ sO.dataSetName+","+( if(sO.trainTestEqual)"t" else "f")+","+
            sO.inferenceMethod+","+sO.dbcfwSeed+","+ (if(sO.dataGenGreyOnly) "t" else "f")+","+ (if(sO.compPerPixLoss) "t" else "f")+","+ sO.dataGenNeighProb+","+ sO.featHistSize+","+
            sO.featCoOcurNumBins+","+ (if(sO.useLoopyBP) "t" else "f")+","+ (if(sO.useMPLP) "t" else "f")+","+ (if(sO.slicNormalizePerClust) "t" else "f")+","+ sO.dataGenOsilNoise+","+ sO.dataRandSeed+","+
            sO.dataGenHowMany+","+sO.slicCompactness+","+( if(sO.putLabelIntoFeat) "t" else "f" )+","+sO.dataAddedNoise+","+(if(sO.modelPairwiseDataDependent) "t" else "f")+","+(if(sO.featIncludeMeanIntensity) "t" else "f")+
            ","+bToS(sO.featAddOffsetColumn)+","+bToS(sO.featAddIntensityVariance)+","+bToS(sO.featNeighHist)+","+ sO.numDataDepGraidBins+","+sO.loopyBPmaxIter+","+bToS(sO.standardizeFeaturesByColumn)+","+bToS(sO.recompFeat)+
            ","+bToS(sO.dataDepUseIntensity)+","+bToS(sO.dataDepUseIntensityByNeighSD)+","+bToS(sO.dataDepUseIntensityBy2NeighSD)+","+bToS(sO.dataDepUseUniqueness)+","+sO.dataDepMeth+","
  
            
            
                
   
    
    
   
    
    val evenMore = (" %.5f, %.5f, %.5f").format(vocScores(0),vocScores(1),(if(vocScores.length>2) vocScores(2) else (-0.0)))+","+sO.lambda+","+labelOccurance(0)+","+labelOccurance(1)+","+(if(labelOccurance.length>2) labelOccurance(2) else (-0.0))
    
    
    
    println("#EndScore#,%d,%s,%s,%d,%.3f,%.3f,%s,%d,%d,%.3f,%s,%d,%d,%s,%s,%d,%s,%f,%f,%d,%s,%s,%.3f,%.3f,%s,%s".format(
        sO.startTime, sO.runName,sO.gitVersion,(0),sO.dataGenSparsity,sO.dataAddedNoise,if(sO.dataNoiseOnlyTest)"t"else"f",sO.dataGenTrainSize,
        sO.dataGenCanvasSize,sO.learningRate,if(sO.useMF)"t"else"f",sO.numClasses,MAX_DECODE_ITERATIONS,if(sO.onlyUnary)"t"else"f",
        if(sO.debug)"t"else"f",sO.roundLimit,if(sO.dataWasGenerated)"t"else"f",avgTestLoss,avgTrainLoss,sO.dataRandSeed , 
        if(sO.useMSRC) "t" else "f", if(sO.useNaiveUnaryMax)"t"else"f" ,avgPerPixTestLoss,avgPerPixTrainLoss, if(sO.trainTestEqual)"t" else "f" , 
        sO.dataSetName )+newLabels+evenMore+","+bToS(sO.featAddSupSize )+","+sO.slicMinBlobSize  ) 
        
  
    

  }

  
}