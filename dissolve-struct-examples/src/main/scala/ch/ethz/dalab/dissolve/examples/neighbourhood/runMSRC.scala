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
 

object runMSRC {

  
  def printSuperPixels(clusterAssign: Array[Array[Array[Int]]], imp2: ImagePlus, borderCol: Double = Int.MaxValue.asInstanceOf[Double], label: String = "NoName") {
    val imp2_pix = new Duplicator().run(imp2);
    val aStack_supPix = imp2_pix.getStack

    val xDim = aStack_supPix.getWidth
    val yDim = aStack_supPix.getHeight
    val zDim = aStack_supPix.getSize

    if (zDim > 5) {
      for (vX <- 0 until xDim) {
        for (vY <- 0 until yDim) {
          var lastLabel = clusterAssign(vX)(vY)(0)
          for (vZ <- 0 until zDim) {
            if (clusterAssign(vX)(vY)(vZ) != lastLabel) {
              aStack_supPix.setVoxel(vX, vY, vZ, borderCol)
            }
            lastLabel = clusterAssign(vX)(vY)(vZ)
          }
        }
      }
    }

    for (vX <- 0 until xDim) {
      for (vZ <- 0 until zDim) {
        var lastLabel = clusterAssign(vX)(0)(vZ)
        for (vY <- 0 until yDim) {

          if (clusterAssign(vX)(vY)(vZ) != lastLabel) {
            aStack_supPix.setVoxel(vX, vY, vZ, borderCol)
          }
          lastLabel = clusterAssign(vX)(vY)(vZ)
        }
      }
    }

    for (vZ <- 0 until zDim) {

      for (vY <- 0 until yDim) {
        var lastLabel = clusterAssign(0)(vY)(vZ)
        for (vX <- 0 until xDim) {
          if (clusterAssign(vX)(vY)(vZ) != lastLabel) {
            aStack_supPix.setVoxel(vX, vY, vZ, borderCol)
          }
          lastLabel = clusterAssign(vX)(vY)(vZ)
        }
      }

    }
    imp2_pix.setStack(aStack_supPix)
    //imp2_pix.show()
    IJ.saveAs(imp2_pix, "bmp", "/home/mort/workspace/Scala-SLIC-Superpixel/ScalaSLIC/data/" + label + "_seg.bmp"); //TODO this should not be hardcoded 
    //TODO free this memory after saving 
  }

  
  def coOccurancePerSuperRGB( mask:Array[Array[Array[Int]]], image: ImageStack, numSuperPix:Int,binsPerColor:Int=3,directions:List[(Int,Int,Int)]=List((1,0,0),(0,1,0),(0,0,1))):Map[Int,Array[Double]]={
        val maxColor = 255
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
        val col = image.getColorModel
        val binWidth = maxColor/binsPerColor
        val maxBin = (binsPerColor-1)*binsPerColor*binsPerColor+(binsPerColor-1)*binsPerColor+(binsPerColor-1)+1
        def whichBin(r:Int,g:Int,b:Int):Int={ 
          val rBin = min(r/binWidth,binsPerColor-1)
          val gBin = min(g/binWidth,binsPerColor-1)
          val bBin = min(b/binWidth,binsPerColor-1)
          rBin*(binsPerColor*binsPerColor)+gBin*binsPerColor+bBin
        }
        //val coOccuranceMaties = new HashMap[Int,Array[Array[Int]]]
        val coOccuranceMaties = (0 until numSuperPix).toList.map { key => (key -> Array.fill(maxBin,maxBin){0}) }.toMap
        val coNormalizingConst = Array.fill(numSuperPix){0.0}
        for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
           val supID = mask(x)(y)(z)
           val rgb= image.getVoxel(x,y,z).asInstanceOf[Int]
           val r=col.getRed(rgb)
           val g=col.getGreen(rgb)
           val b=col.getBlue(rgb)
           val bin = whichBin(r,g,b)
           
           directions.foreach( (el:(Int,Int,Int))=>{
             val neighRGB = image.getVoxel(x+el._1,y+el._2,z+el._3).asInstanceOf[Int]
             val neighBin = whichBin(col.getRed(neighRGB),col.getGreen(neighRGB),col.getBlue(neighRGB))
             coOccuranceMaties.get(supID).get(bin)(neighBin)+=1 //If this throws an error then there must be a super pixel not in (0,numSuperPix]
             coNormalizingConst(supID)+=1.0
             if(bin!=neighBin)
               coOccuranceMaties.get(supID).get(neighBin)(bin)+=1 //I dont care about direction of the neighbour. Someone could also use negative values in the directions 
           })
           
        }
        
        
        //I only need the upper triagular of the matrix since its symetric. Also i want all my features to be vectorized so
        val n = maxBin
 val out=coOccuranceMaties.map( ((pair:(Int,Array[Array[Int]]))=> {
   val key = pair._1
   val myCoOccur = pair._2
   
        
        val linFeat = Array.fill((n * (n+1)) / 2){0.0}; 
    for (r<- 0 until maxBin ; c <- 0 until maxBin)
    {
     
        val i = (n * r) + c-((r * (r+1))/2) 
        linFeat(i) = myCoOccur(r)(c)/coNormalizingConst(key)
      
    }
 (key,linFeat)
 })).toMap
 
   out
  }

  
  def genMSRCsupPix( numClasses : Int, runName:String = "_", isSquare:Boolean=false,splitTraining:Boolean=true,S:Int, imageDataSource:String="../data/generated/msrcSub2/Images", groundTruthDataSource:String = "../data/generated/msrcSub2/GroundTruth"):(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Map[(Int,Int,Int),Int])={
  
  //MSRC_ObjCategImageDatabase_v2
    //msrcSubsetsy
      val rawImgDir =  imageDataSource  
      val groundTruthDir = groundTruthDataSource  
       val lostofBMP = Option(new File(rawImgDir).list).map(_.filter(_.endsWith(".bmp"))).get
       val superType = if(isSquare) "_square_" else "_SLIC_"

         val colorMapPath =  rawImgDir+"/globalColorMap"+".colorLabelMapping"
         val colorMapF = new File(colorMapPath)
       val colorToLabelMap = if(colorMapF.exists()) GraphUtils.readObjectFromFile[HashMap[(Int,Int,Int),Int]](colorMapPath)  else  new HashMap[(Int,Int,Int),Int]()
       
       
       val allData = for( fI <- 0 until lostofBMP.size) yield {
        val fName = lostofBMP(fI)
        val nameNoExt = fName.substring(0,fName.length()-4)
        val rawImagePath =  rawImgDir+"/"+ fName
        
        val graphCachePath = rawImgDir+"/"+ nameNoExt +superType+"_" +runName+".graph"
        val maskpath = rawImgDir+"/"+ nameNoExt +superType+ "_" +runName+".mask"
        val groundCachePath = groundTruthDir+"/"+ nameNoExt+superType+"_" +runName+".ground"
        val perPixLabelsPath = groundTruthDir+"/"+nameNoExt+superType+"_"+runName+".pxground"
        val outLabelsPath = groundTruthDir+"/"+ nameNoExt +superType+"_"+runName+".labels"
        
        val cacheMaskF = new File(maskpath)
        val cacheGraphF = new File(graphCachePath)
        val cahceLabelsF =  new File(outLabelsPath)
        
        if(cacheGraphF.exists() && cahceLabelsF.exists()){//Check if its cached 
          
          val outLabels = GraphUtils.readObjectFromFile[GraphLabels](outLabelsPath)
          val outGraph =  GraphUtils.readObjectFromFile[GraphStruct[breeze.linalg.Vector[Double],(Int,Int,Int)]](graphCachePath)
           new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](outLabels, outGraph)
        } 
        else{
        println("No cache found for "+nameNoExt+" computing now..")
        val tStartPre = System.currentTimeMillis()
        val opener = new Opener();
        val img = opener.openImage(rawImagePath);
        val aStack = img.getStack
        val bitDep = aStack.getBitDepth()
        val colMod = aStack.getColorModel()
        val xDim = aStack.getWidth
        val yDim = aStack.getHeight
        val zDim = aStack.getSize

        val copyImage = Array.fill(xDim, yDim, zDim) { (0, 0, 0) }
        for (x <- 0 until xDim; y <- 0 until yDim; z <- 0 until zDim) {
          val curV = aStack.getVoxel(x, y, z).asInstanceOf[Int]
          copyImage(x)(y)(z) = (colMod.getRed(curV), colMod.getGreen(curV), colMod.getBlue(curV))
        }

        val distFn = (a: (Int, Int, Int), b: (Int, Int, Int)) => sqrt(Math.pow(a._1 - b._1, 2) + Math.pow(a._2 - b._2, 2) + Math.pow(a._3 - b._3, 2))
        val sumFn =  (a: (Int, Int, Int), b: (Int, Int, Int)) => ((a._1 + b._1, a._2 + b._2, a._3 + a._3))
        val normFn = (a: (Int, Int, Int), n: Int)             => ((a._1 / n, a._2 / n, a._3 / n))

        val allGr = new SLIC[(Int, Int, Int)](distFn, sumFn, normFn, copyImage, S, 15, minChangePerIter = 0.002, connectivityOption = "Imperative", debug = false)
        
        val tMask = System.currentTimeMillis()
        val mask = if( cacheMaskF.exists())  {
          GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](maskpath) 
        } else {
          if(isSquare)
             allGr.calcSimpleSquaresSupPix
          else
            allGr.calcSuperPixels 
          }
        println("Calculate SuperPixels time: "+(System.currentTimeMillis()-tMask))
        
       // printSuperPixels(mask, img, 300, nameNoExt+superType+"_mask")//TODO remove this 
        if(!cacheMaskF.exists())
           GraphUtils.writeObjectToFile(maskpath,mask)//save a chace 
       
        
        
        val histBinsPerCol = 5
        val histWidt = 255 / histBinsPerCol
        val featureFn = (data: List[DatumCord[(Int, Int, Int)]]) => {
          val redHist = Array.fill(histBinsPerCol) { 0 }
          val greenHist = (Array.fill(histBinsPerCol) { 0 })
          val blueHist = (Array.fill(histBinsPerCol) { 0 })
          data.foreach(a => {
            redHist(min(histBinsPerCol - 1, (a.cont._1 / histWidt))) += 1
            greenHist(min(histBinsPerCol - 1, (a.cont._2 / histWidt))) += 1
            blueHist(min(histBinsPerCol - 1, (a.cont._3 / histWidt))) += 1

          })
          val all = List(redHist, greenHist, blueHist).flatten
          val mySum = all.sum
          Vector(all.map(a => (a.asInstanceOf[Double] / mySum)).toArray)
        }
        
        
        val tFindCenter = System.currentTimeMillis()
         val (supPixCenter, supPixSize) = allGr.findSupPixelCenterOfMassAndSize(mask)
         println( "Find Center mass time: "+(System.currentTimeMillis()-tFindCenter))
         val tEdges = System.currentTimeMillis()
    val edgeMap = allGr.findEdges_simple(mask, supPixCenter)
    println("Find Graph Connections time: "+(System.currentTimeMillis()-tEdges))
    val tSupBound = System.currentTimeMillis()
   // val (cordWiseBlobs, supPixEdgeCount) = allGr.findSupPixelBounds(mask)
    println ("Group Pixel by Superpixel id: "+(System.currentTimeMillis()-tSupBound))
    val tSupBound_I =System.currentTimeMillis()
    val cordWiseBlobs = allGr.findSupPixelBounds_I(mask)
    println ("Group Pixel by Superpixel id: "+(System.currentTimeMillis()-tSupBound_I))
    
    //TODO REMOVE THIS CHECK 
    //cordWiseBlobs.keySet.foreach { key => assert(cordWiseBlobs.get(key).get.length==cordWiseBlob_I.get(key).get.length) }
    
    //make sure superPixelId's are ascending Ints
    val keys = cordWiseBlobs.keySet.toList.sorted
    //val keymap = (keys.zip(0 until keys.size)).toMap
    assert(keys.equals((0 until keys.size).toList))
    //def k(a: Int): Int = { keymap.get(a).get }

    val tFeat = System.currentTimeMillis()
    
    
    val histFeatures = for (i <- 0 until keys.size) yield { featureFn(cordWiseBlobs.get(i).get) }
    val coOccurFeats = coOccurancePerSuperRGB(mask,aStack,keys.size) //TODO generalize this to work for greyscale data    
    
        
    println("Compute Features per Blob: "+(System.currentTimeMillis()-tFeat))
    val outEdgeMap = edgeMap.map(a => {
      val oldId = a._1
      val oldEdges = a._2
      ((oldId), oldEdges.map { oldEdge => (oldEdge) })
    })
    
    
    val outNodes = for ( id <- 0 until keys.size) yield{
      Node(id,Vector(histFeatures(id).toArray++(coOccurFeats.get(id).get)) , collection.mutable.Set(outEdgeMap.get(id).get.toSeq:_*))
    }
    
    //val linkCords = cordWiseBlobs.map( a=> { a._1->a._2.map(daCord => { (daCord.x,daCord.y,daCord.z)}) })
    val outGraph = new GraphStruct[breeze.linalg.Vector[Double],(Int,Int,Int)](Vector(outNodes.toArray),maskpath) //TODO change the linkCord in graphStruct so it makes sense 
     GraphUtils.writeObjectToFile(graphCachePath,outGraph)
     //Construct Ground Truth 
     val tGround = System.currentTimeMillis()
        val groundTruthpath =  groundTruthDir+"/"+ nameNoExt+"_GT.bmp"
        val openerGT = new Opener();
        val imgGT = openerGT.openImage(groundTruthpath);
        val gtStack = imgGT.getStack
        val gtColMod= gtStack.getColorModel
        assert(xDim==gtStack.getWidth)
        assert(yDim==gtStack.getHeight)
        assert(zDim==gtStack.getSize)
        
        
        val labelCount=   if(colorToLabelMap.keySet.size>0) new AtomicInteger(colorToLabelMap.keySet.size) else new AtomicInteger(0)
        val groundTruthMap = new HashMap[Int,Int]()
        //val groundTruthPerPixel = Array.fill(xDim,yDim,zDim){-1}
        val cordBlob = cordWiseBlobs.iterator
        
        while(cordBlob.hasNext){
          val (pixID,cordD) = cordBlob.next()
          
          val curLabelCount =HashMap[Int,Int]()
          cordD.foreach( datum => {
              val v = gtStack.getVoxel(datum.x,datum.y,datum.z).asInstanceOf[Int]
              
          val r = gtColMod.getRed(v)
          val g = gtColMod.getGreen(v)
          val b = gtColMod.getBlue(v)
          
          if(!colorToLabelMap.contains((r,g,b)))
            colorToLabelMap.put((r,g,b),labelCount.getAndIncrement)
          
          val myCo = colorToLabelMap.get((r,g,b)).get
          val old = curLabelCount.getOrElse(myCo,0)
        //  groundTruthPerPixel(datum.x)(datum.y)(datum.z)=myCo
          curLabelCount.put(myCo,old+1)
          })
          val mostUsedLabel= curLabelCount.maxBy(_._2)._1
          groundTruthMap.put(pixID,mostUsedLabel)
        }
        
        println("Ground Truth Blob Mapping: "+(System.currentTimeMillis()-tGround))
         GraphUtils.writeObjectToFile(groundCachePath,groundTruthMap)
        
     //   GraphUtils.writeObjectToFile( perPixLabelsPath,groundTruthPerPixel)
        val labelsInOrder = (0 until groundTruthMap.size).map(a=>groundTruthMap.get(a).get)
        val outLabels = new GraphLabels(Vector(labelsInOrder.toArray),numClasses,groundTruthpath)
         GraphUtils.writeObjectToFile(outLabelsPath,outLabels)
         
         //TODO remove debugging 
       //  val outError= GraphUtils.lossPerPixel(outGraph.originMapFile, outLabels.originalLabelFile,outLabels,colorMap=colorToLabelMap.toMap)
       //  GraphUtils.printBMPFromGraph(outGraph,outLabels,0,nameNoExt+"_"+"_true_cnst",colorMap=colorToLabelMap.toMap.map(_.swap))
    
          
         println("Total Preprocessing time for "+nameNoExt+" :"+(System.currentTimeMillis()-tStartPre))
         new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](outLabels, outGraph)
        }
        }
      
      
      GraphUtils.writeObjectToFile(colorMapPath,colorToLabelMap)
      val (training,test) =  if(splitTraining) allData.splitAt(allData.size/2) else (allData,allData)
      return (training,test,colorToLabelMap.toMap) 
   }
  
  
  
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
    val MAX_DECODE_ITERATIONS_MF_ALT:Int = options.getOrElse("maxDecodeItrMF",  (MAX_DECODE_ITERATIONS).toString ).toInt
    solverOptions.sample = options.getOrElse("sample", "frac")
    solverOptions.sampleFrac = options.getOrElse("samplefrac", "1").toDouble
    solverOptions.dbcfwSeed = options.getOrElse("dbcfwSeed","-1").toInt
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
    solverOptions.generateMSRCSupPix = options.getOrElse("supMSRC","false").toBoolean
    solverOptions.squareSLICoption = options.getOrElse("supSquareMSRC","false").toBoolean
    solverOptions.trainTestEqual = options.getOrElse("trainTestEqual","false").toBoolean
    solverOptions.superPixelSize = options.getOrElse("S","15").toInt
    solverOptions.dataFilesDir = options.getOrElse("dataDir","###")
    solverOptions.imageDataFilesDir = options.getOrElse("imageDir",if(solverOptions.dataFilesDir=="###") "../data/generated/MSRC_ObjCategImageDatabase_v2/Images" else solverOptions.dataFilesDir+"/Images")
    solverOptions.groundTruthDataFilesDir = options.getOrElse("groundTruthDir",if(solverOptions.dataFilesDir=="###") "../data/generated/MSRC_ObjCategImageDatabase_v2/GroundTruth" else solverOptions.dataFilesDir+"/GroundTruth")
    if(solverOptions.squareSLICoption) 
      solverOptions.generateMSRCSupPix=true
    val DEBUG_COMPARE_MF_FACTORIE =  options.getOrElse("cmpEnergy","false").toBoolean
    
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

    
    //genColorfullSquaresData(howMany: Int, canvasSize: Int, squareSize : Int, portionBackground: Double, numClasses: Int, featureNoise: Double, outputDir:String){
    GraphUtils.genColorfullSquaresDataSuperNoise(50,100,20,0.75,3,0.75,"../data/generated/genDat10",320)
    
    
    def getMSRC():(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Map[(Int, Int, Int), Int])={
          val (oldtrainData, oldtestData) = ImageSegmentationUtils.loadMSRC("../data/generated/MSRC_ObjCategImageDatabase_v2")
      
        val graphTrainD = for (i <- 0 until oldtrainData.size) yield {
          val (gTrain, gLabel) = GraphUtils.convertOT_msrc_toGraph(oldtrainData(i).pattern, oldtrainData(i).label, solverOptions.numClasses)
          new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](gLabel, gTrain)
        }
        val graphTestD = for (i <- 0 until oldtestData.size) yield {
          val (gTrain, gLabel) = GraphUtils.convertOT_msrc_toGraph(oldtestData(i).pattern, oldtestData(i).label, solverOptions.numClasses)
          new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](gLabel, gTrain)
        }
        val colorMapInt = ImageSegmentationUtils.colormapRGB
     
    return (graphTrainD.toArray.toSeq, graphTestD.toArray.toSeq,colorMapInt)
    }
    
    //TODO add features to this noise creator which makes groundTruth files just like those in getMSRC or getMSRCSupPix
    def genSquareNoiseD():(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Map[(Int, Int, Int), Int])= {
      val trainData = GraphUtils.genSquareBlobs(solverOptions.dataGenTrainSize,solverOptions.dataGenCanvasSize,solverOptions.dataGenSparsity,solverOptions.numClasses, if(solverOptions.dataNoiseOnlyTest) 0.0 else solverOptions.dataAddedNoise,solverOptions.dataRandSeed).toArray.toSeq
    val testData = GraphUtils.genSquareBlobs(solverOptions.dataGenTestSize,solverOptions.dataGenCanvasSize,solverOptions.dataGenSparsity,solverOptions.numClasses,solverOptions.dataAddedNoise,solverOptions.dataRandSeed+1).toArray.toSeq

     return (trainData,testData, new HashMap[(Int,Int,Int),Int]().toMap)
    }
    
   
        
    
    
    //TODO add features to this noise creator which makes groundTruth files just like those in getMSRC or getMSRCSupPix
    val (trainData,testData, colorlabelMap) = if(solverOptions.useMSRC) getMSRC() else if(solverOptions.generateMSRCSupPix) {genMSRCsupPix(solverOptions.numClasses,solverOptions.runName,isSquare=solverOptions.squareSLICoption,solverOptions.trainTestEqual,S=solverOptions.superPixelSize,imageDataSource=solverOptions.imageDataFilesDir,groundTruthDataSource= solverOptions.groundTruthDataFilesDir) }else genSquareNoiseD()
   
    
    
    
    /*
    for( i <- 0 until 10){
      val pat = trainData(i).pattern
      val primMat = GraphUtils.flatten3rdDim(GraphUtils.reConstruct3dMat(  trainData(i).label, pat.dataGraphLink,solverOptions.dataGenCanvasSize,solverOptions.dataGenCanvasSize,1))
      println(primMat.deep.mkString("\n"))
      println("------------------------------------------------")
    }
    * 
    */
    
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

        val myGraphSegObj = new GraphSegmentationClass(solverOptions.onlyUnary,MAX_DECODE_ITERATIONS,
            solverOptions.learningRate ,solverOptions.useMF,solverOptions.mfTemp,solverOptions.useNaiveUnaryMax,
            DEBUG_COMPARE_MF_FACTORIE,MAX_DECODE_ITERATIONS_MF_ALT,solverOptions.runName)
    val trainer: StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] =
      new StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels](
        trainDataRDD,
        myGraphSegObj,
        solverOptions)

    val t0MTrain = System.currentTimeMillis()
    val model: StructSVMModel[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] = trainer.trainModel()
    val t1MTrain = System.currentTimeMillis()
    var avgTrainLoss = 0.0
    var avgPerPixTrainLoss = 0.0
    var count=0
    val invColorMap = colorlabelMap.toMap.map(_.swap)
    
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
        GraphUtils.printBMPFromGraph(item.pattern,prediction,0,fileName+solverOptions.runName+"_"+count+"_train_predict",colorMap=invColorMap)
        GraphUtils.printBMPFromGraph(item.pattern,item.label,0,fileName+solverOptions.runName+"_"+count+"_train_true",colorMap=invColorMap)
       
        //val supPixToOrig = GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](item.pattern.originMapFile)
        //val origTrueLabel = GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](item.label.originalLabelFile)
    
      }
     
     avgTrainLoss += myGraphSegObj.lossFn(item.label, prediction) //TODO change this so that it tests the original pixel labels 
     val tFindLoss = System.currentTimeMillis()
     avgPerPixTrainLoss +=  GraphUtils.lossPerPixel(item.pattern.originMapFile, item.label.originalLabelFile,prediction,colorMap=colorlabelMap)
   //  println("PerPixel loss counting"+(System.currentTimeMillis()-tFindLoss))
      count+=1
    }
    avgTrainLoss = avgTrainLoss / trainData.size
    avgPerPixTrainLoss = avgPerPixTrainLoss/trainData.size
    println("\nTRAINING: Avg Loss : " + avgTrainLoss +") PerPix("+avgPerPixTrainLoss+ ") numItems " + trainData.size)
    //Test Error 
    var avgTestLoss = 0.0
    var avgPerPixTestLoss = 0.0
    count=0
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
      avgPerPixTestLoss +=  GraphUtils.lossPerPixel(item.pattern.originMapFile, item.label.originalLabelFile,prediction,colorMap=colorlabelMap)
      avgTestLoss += myGraphSegObj.lossFn(item.label, prediction)
      count+=1
    }
    avgTestLoss = avgTestLoss / testData.size
    avgPerPixTestLoss = avgPerPixTestLoss/ testData.size
    println("\nTest Avg Loss : Sup(" + avgTestLoss +") PerPix("+avgPerPixTestLoss+ ") numItems " + testData.size)
    
    println("#EndScore#,%d,%s,%s,%d,%.3f,%.3f,%s,%d,%d,%.3f,%s,%d,%d,%s,%s,%d,%s,%f,%f,%d,%s,%s".format(
        solverOptions.startTime, solverOptions.runName,solverOptions.gitVersion,(t1MTrain-t0MTrain),solverOptions.dataGenSparsity,solverOptions.dataAddedNoise,if(solverOptions.dataNoiseOnlyTest)"t"else"f",solverOptions.dataGenTrainSize,solverOptions.dataGenCanvasSize,solverOptions.learningRate,if(solverOptions.useMF)"t"else"f",solverOptions.numClasses,MAX_DECODE_ITERATIONS,if(solverOptions.onlyUnary)"t"else"f",if(solverOptions.debug)"t"else"f",solverOptions.roundLimit,if(solverOptions.dataWasGenerated)"t"else"f",avgTestLoss,avgTrainLoss,solverOptions.dataRandSeed , if(solverOptions.useMSRC) "t" else "f", if(solverOptions.useNaiveUnaryMax)"t"else"f"   ) )
        
  
    sc.stop()

  }

}