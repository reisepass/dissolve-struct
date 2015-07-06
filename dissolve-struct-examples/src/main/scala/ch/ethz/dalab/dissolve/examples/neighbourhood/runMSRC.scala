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
import scala.util.Random
import java.awt.image.ColorModel
import java.awt.Color
 

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
    IJ.saveAs(imp2_pix, "tif", "/home/mort/workspace/dissolve-struct/data/supPixImages/" + label + "_seg.tif"); //TODO this should not be hardcoded 
    //TODO free this memory after saving 
  }

  
  def coOccurancePerSuperRGB( mask:Array[Array[Array[Int]]], image: ImageStack, numSuperPix:Int,binsPerColor:Int=3,directions:List[(Int,Int,Int)]=List((1,0,0),(0,1,0),(0,0,1)), maxColorValue:Int=255):Map[Int,Array[Double]]={
        val maxColor = maxColorValue 
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
        val col =  image.getColorModel
        val binWidth = maxColor/binsPerColor
        val maxBin = (binsPerColor-1)*binsPerColor*binsPerColor+(binsPerColor-1)*binsPerColor+(binsPerColor-1)+1
        def whichBin(r:Int,g:Int,b:Int):Int={
          val rBin = min(r/binWidth,binsPerColor-1)
          val gBin = min(g/binWidth,binsPerColor-1)
          val bBin = min(b/binWidth,binsPerColor-1)
          rBin*(binsPerColor*binsPerColor)+gBin*binsPerColor+bBin
        }
        val fillCont = (0 until numSuperPix).toList.map { key => (key -> Array.fill(maxBin,maxBin){0}) }.toMap
        val coOccuranceMaties= new HashMap[Int,Array[Array[Int]]]
        coOccuranceMaties++fillCont
        
        val coNormalizingConst = new HashMap[Int,Double]++((0 until numSuperPix).map(key => (key -> 0.0)).toMap)
        for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
           val supID = mask(x)(y)(z)
           val rgb= image.getVoxel(x,y,z).asInstanceOf[Int]
           val r=col.getRed(rgb)
           val g=col.getGreen(rgb)
           val b=col.getBlue(rgb)
           val bin = whichBin(r,g,b)
           if(!coOccuranceMaties.contains(supID)){
             coOccuranceMaties.put(supID,Array.fill(maxBin,maxBin){0})
             coNormalizingConst.put(supID,0.0)
           }
           directions.foreach( (el:(Int,Int,Int))=>{
             val neighRGB = image.getVoxel(x+el._1,y+el._2,z+el._3).asInstanceOf[Int]
             val neighBin = whichBin(col.getRed(neighRGB),col.getGreen(neighRGB),col.getBlue(neighRGB))
            
             coOccuranceMaties.get(supID).get(bin)(neighBin)+=1 //If this throws an error then there must be a super pixel not in (0,numSuperPix]
             coNormalizingConst(supID)+=1.0
             if(bin!=neighBin){  
               coOccuranceMaties.get(supID).get(neighBin)(bin)+=1 //I dont care about direction of the neighbour. Someone could also use negative values in the directions 
             
             }
             })
           
        }
        
        
        //I only need the upper triagular of the matrix since its symetric. Also i want all my features to be vectorized so

 val out=coOccuranceMaties.map( ((pair:(Int,Array[Array[Int]]))=> {
   val key = pair._1
   val myCoOccur = pair._2
   
        
        val linFeat = Array.fill((maxBin * (maxBin+1)) / 2){0.0}; 
    for (r<- 0 until maxBin ; c <- 0 until maxBin)
    {
     
        val i = (maxBin * r) + c-((r * (r+1))/2) 
        linFeat(i) = myCoOccur(r)(c)/coNormalizingConst(key)
      
    }
    val normFeat = normalize(DenseVector(linFeat))
 (key,normFeat.toArray)
 })).toMap
 
   out
  }
  
    def colorhist(image:ImageStack,mask:Array[Array[Array[Int]]],histBinsPerCol:Int,histWidt:Int, maxColorValue:Int=255):Map[Int,Array[Double]]={
     val colMod = image.getColorModel()
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
     val supPixMap = new HashMap[Int,(Array[Int],Array[Int],Array[Int])]
          
       //   val redHist = Array.fill(histBinsPerCol) { 0 }
         // val greenHist = (Array.fill(histBinsPerCol) { 0 })
          //val blueHist = (Array.fill(histBinsPerCol) { 0 })
          
          for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
            val lab = mask(x)(y)(z)
            val refHist = supPixMap.getOrElseUpdate(lab, (Array.fill(histBinsPerCol) { 0 },Array.fill(histBinsPerCol) { 0 },Array.fill(histBinsPerCol) { 0 }))
            val col = image.getVoxel(x,y,z).asInstanceOf[Int]
            val r = colMod.getRed(col)
            val g = colMod.getGreen(col)
            val b = colMod.getBlue(col)
            refHist._1(min(histBinsPerCol - 1, (r / histWidt))) += 1
            refHist._2(min(histBinsPerCol - 1, (g / histWidt))) += 1
            refHist._3(min(histBinsPerCol - 1, (b / histWidt))) += 1
        }
          val keys = supPixMap.keySet.toList.sorted
          keys.map { key => {
            val hists = supPixMap.get(key).get
            val all = List(hists._1,hists._2,hists._3).flatten
            val mySum = all.sum
            (key-> all.map(a => (a.asInstanceOf[Double] / mySum)).toArray)
          }
          }.toMap 
          
   }
   
   def greyHist  (image:ImageStack,mask:Array[Array[Array[Int]]],histBinsPerCol:Int,histWidt:Int, maxColorValue:Int=255):Map[Int,Array[Double]]={
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
     val supPixMap = new HashMap[Int,Array[Int]] 
         for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
            val lab = mask(x)(y)(z)
            val refHist = supPixMap.getOrElseUpdate(lab, (Array.fill(histBinsPerCol) { 0 }))
            val col = image.getVoxel(x,y,z).asInstanceOf[Int]
        
            refHist(min(histBinsPerCol - 1, (col / histWidt))) += 1
        }
          val keys = supPixMap.keySet.toList.sorted
        val out =  keys.map { key => {
            val hist = supPixMap.get(key).get.toList
            val mySum = hist.sum
            (key-> hist.map(a => (a.asInstanceOf[Double] / mySum)).toArray)
          }
          }.toMap
    
        out
   }
   
   def greyCoOccurancePerSuper(image:ImageStack,mask:Array[Array[Array[Int]]],histBins:Int,directions:List[(Int,Int,Int)]=List((1,0,0),(0,1,0),(0,0,1)), maxColorValue:Int=255):Map[Int,Array[Double]]={
        val bitDepth = image.getBitDepth
        assert(bitDepth==8)//TODO can i generalize this to any bit depth ?
        val maxAbsValue = maxColorValue //Divide by 2 because of sign 
        //TODO this is most likely wrong 
        
        val binWidth = maxAbsValue/histBins.asInstanceOf[Double]
        val whichBin = (a:Int)=>{ (a/binWidth).asInstanceOf[Int]}
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
        val gCoOcMats = new HashMap[Int,Array[Array[Int]]]
        val coNormalizingConst = new HashMap[Int,Int]
           for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
             
             val lab=mask(x)(y)(z)
             val myMat = gCoOcMats.getOrElseUpdate(lab, Array.fill(histBins,histBins){0})
             val myCol=image.getVoxel(x,y,z).asInstanceOf[Int]
             val myBin = whichBin(myCol)
             directions.foreach( (dir:(Int,Int,Int))=>{
               val neighCol = image.getVoxel(x+dir._1,y+dir._2,z+dir._3).asInstanceOf[Int]
               val neighBin = whichBin(neighCol)
               val oldZ = coNormalizingConst.getOrElse(lab,0)
               coNormalizingConst.put(lab,oldZ+1)
               myMat(myBin)(neighBin)+=1
               if(myBin!=neighBin)
                 myMat(neighBin)(myBin)+=1
               
             })
             
             
           }
   //I only need the upper triagular of the matrix since its symetric. Also i want all my features to be vectorized so
   val n = histBins
   val out=gCoOcMats.map( ((pair:(Int,Array[Array[Int]]))=> {
   val key = pair._1
   val myCoOccur = pair._2
   
        
        val linFeat = Array.fill((n * (n+1)) / 2){0.0}; 
    for (r<- 0 until n ; c <- 0 until n)
    {
     
        val i = (n * r) + c-((r * (r+1))/2) 
        linFeat(i) = myCoOccur(r)(c)/coNormalizingConst(key)
      
    }
    val normFeat = normalize(DenseVector(linFeat.toArray))
 (key,normFeat.toArray)
 })).toMap
        
     out
   }

 
   
   def genMSRCsupPixV2 ( numClasses:Int,S:Int, imageDataSource:String, groundTruthDataSource:String,  featureFn:(ImageStack,Array[Array[Array[Int]]])=>Map[Int,Array[Double]] ,randomSeed:Int =(-1), runName:String = "_", isSquare:Boolean=false,doNotSplit:Boolean=false, debugLabelInFeat:Boolean=false):(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Map[Int,Int],Map[Int,Double], Array[Array[Double]])={
    
    //distFn:((Double,Double)=>Double),sumFn:(Double,Double)=>Double,normFn:(Double,Int)=>Double, //TODO remove me
     val random = if(randomSeed==(-1)) new Random() else new Random(randomSeed)
      val rawImgDir =  imageDataSource  
      val groundTruthDir = groundTruthDataSource  
       val lotsofBMP = Option(new File(rawImgDir).list).map(_.filter(_.endsWith(".bmp")))
       val lotsofTIF = Option(new File(rawImgDir).list).map(_.filter(_.endsWith(".tif")))
       val lotsofTIFF = lotsofTIF ++ Option(new File(rawImgDir).list).map(_.filter(_.endsWith(".tiff")))
       val allFiles =( lotsofTIFF ++ lotsofBMP).flatten.toArray
       val extension=if(lotsofBMP.get.size>0 ){
         assert(lotsofTIFF.flatten.size==0,"Currently we only support uniform datatypes among the training and testing images. Dir searched:"+rawImgDir )
         ".bmp"
       }
       else if(lotsofTIFF.size>0){
         assert(lotsofBMP.get.size==0,"Currently we only support uniform datatypes among the training and testing images. Dir searched:"+rawImgDir )
         ".tif"
       }
       else
         ".tif"
       
         
       
       val superType = if(isSquare) "_square_" else "_SLIC_"

         val colorMapPath =  rawImgDir+"/globalColorMap"+".colorLabelMapping2"
         val colorMapF = new File(colorMapPath)
         val colorToLabelMap = if(colorMapF.exists()) GraphUtils.readObjectFromFile[HashMap[Int,Int]](colorMapPath)  else  new HashMap[Int,Int]()
         val classCountPath = rawImgDir+"/classCountMap"+superType+S+"_" +runName+".classCount"
         val classCountF = new File(classCountPath)
         val oldClassCountFound = classCountF.exists()
         val totalClassCount = if(oldClassCountFound) GraphUtils.readObjectFromFile[HashMap[Int,Double]](classCountPath) else new HashMap[Int,Double]()
     
         val transProbPath = rawImgDir+"/transitionProbabilityMatrix"+superType+S+"_" +runName+".transProb"
         val transProbF = new File(transProbPath)
         val transProbFound = transProbF.exists()
         val transProb = if(transProbFound) GraphUtils.readObjectFromFile[Array[Array[Double]]](transProbPath) else Array.fill(numClasses,numClasses){0.0}
         var totalConn= if(transProbFound) 1.0 else 0.0
         
        
        assert(allFiles.size>0)
        val allData = for( fI <- 0 until allFiles.size) yield {
        val fName = allFiles(fI)
        val nameNoExt = fName.substring(0,fName.length()-4)
        val rawImagePath =  rawImgDir+"/"+ fName
        
        val graphCachePath = rawImgDir+"/"+ nameNoExt +superType+S+"_" +runName+".graph2"
        val maskpath = rawImgDir+"/"+ nameNoExt +superType+ S+"_" +runName+".mask"
        val groundCachePath = groundTruthDir+"/"+ nameNoExt+superType+S+"_" +runName+".ground2"
        val perPixLabelsPath = groundTruthDir+"/"+nameNoExt+superType+S+"_"+runName+".pxground2"
        val outLabelsPath = groundTruthDir+"/"+ nameNoExt +superType+S+"_"+runName+".labels2"
        
        val cacheMaskF = new File(maskpath)
        val cacheGraphF = new File(graphCachePath)
        val cahceLabelsF =  new File(outLabelsPath)
        var imgSrcBidDepth=(-1)
        

        
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
        if(imgSrcBidDepth==(-1))
          imgSrcBidDepth=bitDep
        if(imgSrcBidDepth!=(-1))
          assert(bitDep==imgSrcBidDepth,"All images need to have the same bitDepth")
        val isColor = if(bitDep==8) false else true //TODO mabye this is not the best way to check for color in the image 
        val colMod = aStack.getColorModel()
        val xDim = aStack.getWidth
        val yDim = aStack.getHeight
        val zDim = aStack.getSize
        
        
        

        val copyImage = Array.fill(xDim, yDim, zDim) { 0}
        var largest = 0;
        var smallest = 0;
        for (x <- 0 until xDim; y <- 0 until yDim; z <- 0 until zDim) {
          copyImage(x)(y)(z) = aStack.getVoxel(x, y, z).asInstanceOf[Int] //TODO this may be causing an inaccuracy on the image color. Question is why is Imagej outputing Doubles....
          if(copyImage(x)(y)(z)>largest)
            largest=copyImage(x)(y)(z)
          if(copyImage(x)(y)(z)<smallest)
            smallest = copyImage(x)(y)(z)
        }
        
        val distFnCol = (a: (Int, Int, Int), b: (Int, Int, Int)) => sqrt(Math.pow(a._1 - b._1, 2) + Math.pow(a._2 - b._2, 2) + Math.pow(a._3 - b._3, 2))
        val sumFnCol =  (a: (Int, Int, Int), b: (Int, Int, Int)) => ((a._1 + b._1, a._2 + b._2, a._3 + a._3))
        val normFnCol = (a: (Int, Int, Int), n: Int)             => {
          
                   
          ((a._1 / n, a._2 / n, a._3 / n))}

        val distFn = if(isColor) { 
          ( a:Int, b:Int) => distFnCol((colMod.getRed(a.asInstanceOf[Int]),colMod.getGreen(a.asInstanceOf[Int]),colMod.getBlue(a.asInstanceOf[Int])),(colMod.getRed(b.asInstanceOf[Int]),colMod.getGreen(b.asInstanceOf[Int]),colMod.getBlue(b.asInstanceOf[Int])))  
        } else {
          (a:Int,b:Int)=> sqrt(Math.pow(a-b,2))
        }
        
        val rollingAvgFn = if(isColor){
          (a:Int,b:Int,n:Int)=>{
            
            val totalSum= sumFnCol((colMod.getRed(a)*(n-1),colMod.getGreen(a)*(n-1),colMod.getBlue(a)*(n-1)),(colMod.getRed(b),colMod.getGreen(b),colMod.getBlue(b)))
            val cAvg = normFnCol(totalSum,n)
            new Color(cAvg._1,cAvg._2,cAvg._3).getRGB()
          }
        }
        else{
          (a:Int,b:Int,n:Int)=>{
            val sum = a*n + b
            (sum/(n+1)).asInstanceOf[Int]
          }
        }
        
        val sumFn = if(isColor) 
          (a:Int,b:Int) => {
            
         val c= sumFnCol((colMod.getRed(a),colMod.getGreen(a),colMod.getBlue(a)),(colMod.getRed(b),colMod.getGreen(b),colMod.getBlue(b)))
         new Color(c._1,c._2,c._3).getRGB() //TODO I bet this is not the same encoding ImageJ uses
           }else
          (a:Int,b:Int)=> a+b
        
        val normFn = if(isColor)
          (a:Int,n:Int) => {
            val c= normFnCol( (colMod.getRed(a.asInstanceOf[Int]),colMod.getGreen(a.asInstanceOf[Int]),colMod.getBlue(a.asInstanceOf[Int])),n)
            new Color(c._1,c._2,c._3).getRGB()  //TODO I bet this is not the same encoding ImageJ uses
          }
          else
            (a:Int,n:Int) => round(a/n)
          
          
        val allGr =  new SLIC[Int](distFn, rollingAvgFn, normFn, copyImage, S, 15, minChangePerIter = 0.002, connectivityOption = "Imperative", debug = false)   
        
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
        printSuperPixels(mask,img,300,"_supPix_"+fI)//TODO remove me 
          if(!cacheMaskF.exists())
           GraphUtils.writeObjectToFile(maskpath,mask)//save a chace 
       
             val tFindCenter = System.currentTimeMillis()
         val (supPixCenter, supPixSize) = allGr.findSupPixelCenterOfMassAndSize(mask)
         println( "Find Center mass time: "+(System.currentTimeMillis()-tFindCenter))
         val tEdges = System.currentTimeMillis()
    val edgeMap = allGr.findEdges_simple(mask, supPixCenter)
    println("Find Graph Connections time: "+(System.currentTimeMillis()-tEdges))

    
    
    
    //TODO create an equivalent test for this V2
    val keys = supPixCenter.keySet.toList.sorted
    assert(keys.equals((0 until keys.size).toList),"The Superpixel id counter skipped something "+keys.mkString(","))
  
    val outEdgeMap = edgeMap.map(a => {
      val oldId = a._1
      val oldEdges = a._2
      ((oldId), oldEdges.map { oldEdge => (oldEdge) })
    })
    
    
    
     /*
      * 
      * 
      * 
      */
     //Construct Ground Truth 
     val tGround = System.currentTimeMillis()
        val groundTruthpath =  groundTruthDir+"/"+ nameNoExt+"_GT"+extension
        val openerGT = new Opener();
        val imgGT = openerGT.openImage(groundTruthpath);
        val gtStack = imgGT.getStack
        assert(xDim==gtStack.getWidth)
        assert(yDim==gtStack.getHeight)
        assert(zDim==gtStack.getSize)
        
        
        val labelCount=   if(colorToLabelMap.keySet.size>0) new AtomicInteger(colorToLabelMap.keySet.size) else new AtomicInteger(0)
      
        
        val labelCountsPerS =  Array.fill(keys.size){ new HashMap[Int,Int]()}
        val topCountPerS =  Array.fill(keys.size){ 0} 
        val topLabelPerS = Array.fill(keys.size){-1}
        
        for ( x <- 0 until xDim; y<- 0 until yDim; z<- 0 until zDim){
          val spxid = mask(x)(y)(z)
          val col = gtStack.getVoxel(x,y,z).asInstanceOf[Int]
          val lab = colorToLabelMap.getOrElse(col, { val newLab = labelCount.getAndIncrement; colorToLabelMap.put(col,newLab); newLab })
          val lastCount = labelCountsPerS(spxid).getOrElse(lab,0)
          labelCountsPerS(spxid).put(lab,lastCount+1)
          if( (lastCount+1)>topCountPerS(spxid)){
            topCountPerS(spxid)=lastCount+1
            topLabelPerS(spxid)=lab
          }
          
          if(!oldClassCountFound){
            val oldC = totalClassCount.getOrElse(lab,0.0)
            totalClassCount.put(lab,oldC+1.0)
          }
            
        }
          val groundTruthMap =  ( (0 until keys.size) zip topLabelPerS).toMap
        
        
        println("Ground Truth Blob Mapping: "+(System.currentTimeMillis()-tGround))
         GraphUtils.writeObjectToFile(groundCachePath,groundTruthMap)
        
          val tFeat = System.currentTimeMillis()
    val featureVectors = if(debugLabelInFeat){
      
      val tmp = for(id <-0 until keys.size) yield{
       val lab=  groundTruthMap.get(id).get
       val feat = Array.fill(1){0.0}
       feat(0)=lab.asInstanceOf[Double]
       (id, feat)
      }
       tmp.toMap
      
      
    }
    else{
      featureFn(aStack,mask)
    }
   
          println("Compute Features per Blob: "+(System.currentTimeMillis()-tFeat))
    val maxId = max(supPixCenter.keySet)
    val outNodes = for ( id <- 0 until keys.size) yield{
      Node(id,Vector(featureVectors.get(id).get) , collection.mutable.Set(outEdgeMap.get(id).get.toSeq:_*))
        }
    val outGraph = new GraphStruct[breeze.linalg.Vector[Double],(Int,Int,Int)](Vector(outNodes.toArray),maskpath) //TODO change the linkCord in graphStruct so it makes sense
     GraphUtils.writeObjectToFile(graphCachePath,outGraph)    
  
         
         
        val labelsInOrder = (0 until groundTruthMap.size).map(a=>groundTruthMap.get(a).get)
        val outLabels = new GraphLabels(Vector(labelsInOrder.toArray),numClasses,groundTruthpath)
         GraphUtils.writeObjectToFile(outLabelsPath,outLabels)
         
              if(!transProbFound){
            
            edgeMap.foreach((a:(Int,Set[Int]))=>{
              val leftN = a._1
              val leftClass = labelsInOrder(leftN)
              val neigh = a._2
             
              neigh.foreach { thisNeigh => {
                val rightClass = labelsInOrder(thisNeigh)
                transProb(leftClass)(rightClass)+=1.0
                transProb(rightClass)(leftClass)+=1.0
                totalConn+=2.0
              } }
            })
            
          }
         
         //TODO remove debugging 
        GraphUtils.printSupPixLabels_Int(outGraph,outLabels,0,nameNoExt+"_"+"_idMap_",colorMap=colorToLabelMap.toMap.map(_.swap))
        //GraphUtils.printBMPFromGraphInt(outGraph,outLabels,0,nameNoExt+"_"+"_true_cnst",colorMap=colorToLabelMap.toMap.map(_.swap))
    
          
         println("Total Preprocessing time for "+nameNoExt+" :"+(System.currentTimeMillis()-tStartPre))
         new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](outLabels, outGraph)
        
       }
     
       }
     //loss greater than 1
     
      assert(colorToLabelMap.size ==numClasses)
      if(colorToLabelMap.size!=numClasses){
        println("[WARNING] The input numClasses"+numClasses+" is not equal to the number of distinct colors found in the ground truth "+colorToLabelMap.size)
      }
      GraphUtils.writeObjectToFile(colorMapPath,colorToLabelMap)
      
      val classFreq = if(!oldClassCountFound){
        val totalSupP = totalClassCount.values.sum 
        val classFreq = totalClassCount.map( (a:(Int,Double))=>{
          a._1 -> a._2/totalSupP
        })
        GraphUtils.writeObjectToFile(classCountPath,classFreq)
        classFreq
      }
      else
        totalClassCount
        
      val outTransProb = if(transProbFound) transProb else {
        assert(totalConn>0.0)
        for( l <- 0 until numClasses; r<- 0 until numClasses){
        transProb(l)(r)=transProb(l)(r)/totalConn
      }
          GraphUtils.writeObjectToFile(transProbPath,transProb)
        transProb
      }
   
      val shuffleIdx = random.shuffle((0 until allData.size).toList)
      val (trainIDX,testIDX) = if(!doNotSplit) shuffleIdx.splitAt(round(allData.size/2)) else ((0 until allData.size).toList,(0 until allData.size).toList)
      val  training = trainIDX.map { idx => allData(idx) }.toIndexedSeq
      val  test = testIDX.map{idx => allData(idx)}.toIndexedSeq
       println("First Image has "+training(0).pattern.size +" superPixels")
      return (training,test,colorToLabelMap.toMap,classFreq.toMap,outTransProb)
  }
     
  
  /*
  def genMSRCsupPix( numClasses : Int, runName:String = "_", isSquare:Boolean=false,doNotSplit:Boolean=false,S:Int, imageDataSource:String, groundTruthDataSource:String, randomSeed:Int =(-1)):(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Map[(Int,Int,Int),Int])={
  
  //MSRC_ObjCategImageDatabase_v2
    //msrcSubsetsy
    val random = if(randomSeed==(-1)) new Random() else new Random(randomSeed)
      val rawImgDir =  imageDataSource  
      val groundTruthDir = groundTruthDataSource  
       val lotsofBMP = Option(new File(rawImgDir).list).map(_.filter(_.endsWith(".bmp")))
       val lotsofTIF = Option(new File(rawImgDir).list).map(_.filter(_.endsWith(".tif")))
       val lotsofTIFF = lotsofTIF ++ Option(new File(rawImgDir).list).map(_.filter(_.endsWith(".tiff")))
       val allFiles =( lotsofTIFF ++ lotsofBMP).flatten.toArray
       val extension=if(lotsofBMP.isDefined){
         assert(lotsofTIFF.flatten.size==0,"Currently we only support uniform datatypes among the training and testing images. Dir searched:"+rawImgDir )
         ".bmp"
       }
       else if(lotsofTIFF.size>0){
         assert(!lotsofBMP.isDefined,"Currently we only support uniform datatypes among the training and testing images. Dir searched:"+rawImgDir )
         ".tif"
       }
       else
         ".tif"
       
         
       
       val superType = if(isSquare) "_square_" else "_SLIC_"

         val colorMapPath =  rawImgDir+"/globalColorMap"+".colorLabelMapping"
         val colorMapF = new File(colorMapPath)
       val colorToLabelMap = if(colorMapF.exists()) GraphUtils.readObjectFromFile[HashMap[(Int,Int,Int),Int]](colorMapPath)  else  new HashMap[(Int,Int,Int),Int]()
       
       
       val allData = for( fI <- 0 until allFiles.size) yield {
        val fName = allFiles(fI)
        val nameNoExt = fName.substring(0,fName.length()-4)
        val rawImagePath =  rawImgDir+"/"+ fName
        
        val graphCachePath = rawImgDir+"/"+ nameNoExt +superType+S+"_" +runName+".graph"
        val maskpath = rawImgDir+"/"+ nameNoExt +superType+ S+"_" +runName+".mask"
        val groundCachePath = groundTruthDir+"/"+ nameNoExt+superType+S+"_" +runName+".ground"
        val perPixLabelsPath = groundTruthDir+"/"+nameNoExt+superType+S+"_"+runName+".pxground"
        val outLabelsPath = groundTruthDir+"/"+ nameNoExt +superType+S+"_"+runName+".labels"
        
        val cacheMaskF = new File(maskpath)
        val cacheGraphF = new File(graphCachePath)
        val cahceLabelsF =  new File(outLabelsPath)
        var imgSrcBidDepth=(-1)
        
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
        if(imgSrcBidDepth==(-1))
          imgSrcBidDepth=bitDep
        if(imgSrcBidDepth!=(-1))
          assert(bitDep==imgSrcBidDepth,"All images need to have the same bitDepth")
        val isColor = if(bitDep==8) false else true //TODO mabye this is not the best way to check for color in the image 
        val colMod = aStack.getColorModel()
        val xDim = aStack.getWidth
        val yDim = aStack.getHeight
        val zDim = aStack.getSize

        val copyImage = Array.fill(xDim, yDim, zDim) { (0, 0, 0) }
        for (x <- 0 until xDim; y <- 0 until yDim; z <- 0 until zDim) {
          val curV = aStack.getVoxel(x, y, z).asInstanceOf[Int]
          copyImage(x)(y)(z) = (colMod.getRed(curV), colMod.getGreen(curV), colMod.getBlue(curV))
        }

        val distFnCol = (a: (Int, Int, Int), b: (Int, Int, Int)) => sqrt(Math.pow(a._1 - b._1, 2) + Math.pow(a._2 - b._2, 2) + Math.pow(a._3 - b._3, 2))
        val sumFnCol =  (a: (Int, Int, Int), b: (Int, Int, Int)) => ((a._1 + b._1, a._2 + b._2, a._3 + a._3))
        val normFnCol = (a: (Int, Int, Int), n: Int)             => ((a._1 / n, a._2 / n, a._3 / n))
        
        val distFnGrey = (a:Int,b:Int)=> sqrt(Math.pow(a-b,2))
        val sumFnGrey = (a:Int,b:Int) => a+b
        val normFnGrey = (a:Int, n:Int)=> a/n

        
        val allGr =  new SLIC[(Int, Int, Int)](distFnCol, sumFnCol, normFnCol, copyImage, S, 15, minChangePerIter = 0.002, connectivityOption = "Imperative", debug = false)   
        
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
    assert(keys.equals((0 until keys.size).toList),"The Superpixel id counter skipped something "+keys.mkString(","))
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
        val groundTruthpath =  groundTruthDir+"/"+ nameNoExt+"_GT"+extension
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
      
      assert(colorToLabelMap.size <=numClasses)
      if(colorToLabelMap.size!=numClasses){
        println("[WARNING] The input numClasses"+numClasses+" is not equal to the number of distinct colors found in the ground truth "+colorToLabelMap.size)
      }
      GraphUtils.writeObjectToFile(colorMapPath,colorToLabelMap)
      
      val shuffleIdx = random.shuffle((0 until allData.size).toList)
      val (trainIDX,testIDX) = if(!doNotSplit) shuffleIdx.splitAt(round(allData.size/2)) else ((0 until allData.size).toList,(0 until allData.size).toList)
      val  training = trainIDX.map { idx => allData(idx) }.toIndexedSeq
      val  test = testIDX.map{idx => allData(idx)}.toIndexedSeq
      return (training,test,colorToLabelMap.toMap) 
   }
  */
  
  
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
    
    /*
    val path1 = "/home/mort/workspace/dissolve-struct/data/generated/neuro2/Images/"
    val path2 = "/home/mort/workspace/dissolve-struct/data/generated/neuro2/GroundTruth/"
    val mask40 = GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](path1+"training2_SLIC_20_none.mask")
     val opener = new Opener();
        val img = opener.openImage(path1+"training2.tif");

        printSuperPixels(mask40,img,150,"_s_20_neuroTrain_")
        
                val img2 = opener.openImage(path2+"training2_GT.tif");
    
            printSuperPixels(mask40,img2,150,"_s_20_neuroTrue_")
    
    */
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
    solverOptions.useClassFreqWeighting =  options.getOrElse("weightedClassFreq","false").toBoolean
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
    val GEN_NEW_SQUARE_DATA=options.getOrElse("GEN_NEW_SQUARE_DATA","false").toBoolean
    solverOptions.isColor = options.getOrElse("isColor","true").toBoolean
    solverOptions.superPixelSize = options.getOrElse("S","15").toInt
    solverOptions.initWithEmpiricalTransProb = options.getOrElse("initTransProb","false").toBoolean
    solverOptions.dataFilesDir = options.getOrElse("dataDir","###")
    assert(!solverOptions.dataFilesDir.equals("###"),"Error,  Please specify a the 'dataDir' paramater")
    val tmpDir = solverOptions.dataFilesDir.split("/")
    solverOptions.dataSetName = tmpDir(tmpDir.length-1)
    solverOptions.imageDataFilesDir = options.getOrElse("imageDir",  solverOptions.dataFilesDir+"/Images")
    solverOptions.groundTruthDataFilesDir = options.getOrElse("groundTruthDir",  solverOptions.dataFilesDir+"/GroundTruth")
    solverOptions.weighDownPairwise = options.getOrElse("weighDownPairwise","1.0").toDouble
    assert(solverOptions.weighDownPairwise<=1.0 &&solverOptions.weighDownPairwise>=0)
    solverOptions.weighDownUnary = options.getOrElse("weighDownUnary","1.0").toDouble
    assert(solverOptions.weighDownUnary<=1.0&&solverOptions.weighDownUnary>=0)
    if(solverOptions.squareSLICoption) 
      solverOptions.generateMSRCSupPix=true
    val DEBUG_COMPARE_MF_FACTORIE =  options.getOrElse("cmpEnergy","false").toBoolean
    
    if(solverOptions.dataGenSparsity> 0)
      solverOptions.dataWasGenerated=true
    solverOptions.LOSS_AUGMENTATION_OVERRIDE = options.getOrElse("LossAugOverride", "false").toBoolean
    solverOptions.inferenceMethod= if(solverOptions.useMF) "MF" else if(solverOptions.useNaiveUnaryMax) "NAIVE_MAX" else "Factorie"
    solverOptions.putLabelIntoFeat = options.getOrElse("labelInFeat","false").toBoolean
    
    
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
    if(GEN_NEW_SQUARE_DATA){
           val lotsofBMP = Option(new File(solverOptions.dataFilesDir+"/Images").list).map(_.filter(_.endsWith(".bmp")))
           assert(lotsofBMP.isEmpty,"You tried to create new data into a folder which has alreayd been used for a previous dataset")
        GraphUtils.genColorfullSquaresDataSuperNoise(20,150,10,0.5,solverOptions.numClasses,0.0,0.0,solverOptions.dataFilesDir,solverOptions.dataRandSeed)
    }
    
    val runCFG=Option(new File(solverOptions.dataFilesDir+"/"+solverOptions.runName+"_run.cfg")).get
    if(!runCFG.exists()){
      val pw = new PrintWriter(new File(solverOptions.dataFilesDir+"/"+solverOptions.runName+"_run.cfg"))
      pw.write(options.toString)
      pw.write("\n")
      pw.write(solverOptions.toString())
      pw.close
    }
    else{
      val pw = new PrintWriter(new File(solverOptions.dataFilesDir+"/"+solverOptions.runName+"_"+System.currentTimeMillis()+"_run.cfg"))
      pw.write(options.toString)
      pw.write("\n")
      pw.write(solverOptions.toString())
      pw.close
    }
       
    
    
    
    
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
    
   
      //  featureFn:(ImageStack,Array[Array[Array[Int]]])=>Map[Int,Array[Double]] 
    //TODO REMOVE DEBUG new method 
   val histBinsPerCol = 4
   val histBinsPerGray = 8
   val histWidt = 255 / histBinsPerCol
   //BOOKMARK
   
 
   
   val featFn2 = (image:ImageStack,mask:Array[Array[Array[Int]]])=>{
     val xDim = mask.length
     val yDim = mask(0).length
     val zDim = mask(0)(0).length
     val numSupPix = mask(xDim-1)(yDim-1)(zDim-1)+5 //TODO is this correct always ?
      val bitDep = image.getBitDepth()
        val isColor = if(bitDep==8) false else true //TODO maybe this is not the best way to check for color in the image
        //TODO the bit depth should give me the max value which the hist should span over 
        
        if(isColor){
          val hist=colorhist(image,mask,histBinsPerCol,255 / histBinsPerCol)
          val coMat = coOccurancePerSuperRGB(mask, image, numSupPix, histBinsPerCol)
          hist++coMat
        }
       else{
          val hist=greyHist(image,mask,histBinsPerGray,255 / (histBinsPerGray))
          //val coMat= greyCoOccurancePerSuper(image, mask, histBinsPerCol)
          hist //++coMat
       }      
   }
  // 
    
    
    //TODO add features to this noise creator which makes groundTruth files just like those in getMSRC or getMSRCSupPix
    val (trainData,testData, colorlabelMap, classFreqFound,transProb) = genMSRCsupPixV2(solverOptions.numClasses, solverOptions.superPixelSize, solverOptions.imageDataFilesDir, solverOptions.groundTruthDataFilesDir, featFn2, solverOptions.dbcfwSeed, solverOptions.runName, solverOptions.squareSLICoption, solverOptions.trainTestEqual,solverOptions.putLabelIntoFeat) 
   
    
    
    
    println("Train SgreyHistize:"+trainData.size)
    println("Test Size:"+testData.size)
    
    
    //TODO debug remove
    
    println("------ First couple of features ------")
    val pat = trainData(0).pattern
    val name =  trainData(0).pattern.originMapFile
    println(name)
    for(i <- 0 until pat.size){
      println("F<"+pat.getF(i)+">")
    }
    
    
    
     if(solverOptions.initWithEmpiricalTransProb){
      val featureSize =trainData(0).pattern.getF(0).size
      val unaryWeight =Array.fill(solverOptions.numClasses*featureSize){0.0}
      val pairwiseWeight =  (new DenseVector(transProb.flatten)):/(10.0)
      val initWeight = unaryWeight++pairwiseWeight.toArray
      solverOptions.initWeight=initWeight
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
    
    if(solverOptions.useMSRC) solverOptions.numClasses = 24


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
            DEBUG_COMPARE_MF_FACTORIE,MAX_DECODE_ITERATIONS_MF_ALT,solverOptions.runName,
            if(solverOptions.useClassFreqWeighting) classFreqFound else null,
            solverOptions.weighDownUnary,solverOptions.weighDownPairwise, solverOptions.LOSS_AUGMENTATION_OVERRIDE)
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
        GraphUtils.printBMPFromGraphInt(item.pattern,prediction,0,fileName+solverOptions.runName+"_"+count+"-train_predict",colorMap=invColorMap)
        GraphUtils.printBMPFromGraphInt(item.pattern,item.label,0,fileName+solverOptions.runName+"_"+count+"-train_true",colorMap=invColorMap)
       
        //val supPixToOrig = GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](item.pattern.originMapFile)
        //val origTrueLabel = GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](item.label.originalLabelFile)
    
      }
     
     avgTrainLoss += myGraphSegObj.lossFn(item.label, prediction) //TODO change this so that it tests the original pixel labels 
     val tFindLoss = System.currentTimeMillis()
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
      val labelOccurance = Vector(Array.fill(solverOptions.numClasses){0.0})
    val labelIncorrectlyUsed = Vector(Array.fill(solverOptions.numClasses){0.0})
    val labelNotRecog = Vector(Array.fill(solverOptions.numClasses){0.0})
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
    val labelOccurance = Vector(Array.fill(solverOptions.numClasses){0.0})
    val labelIncorrectlyUsed = Vector(Array.fill(solverOptions.numClasses){0.0})
    val labelNotRecog = Vector(Array.fill(solverOptions.numClasses){0.0})
    
    
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
    println("\nTest Avg Loss : Sup(" + avgTestLoss +") PerPix("+avgPerPixTestLoss+ ") numItems " + testData.size)
    println()
    println("Occurances:           \t"+labelOccurance)
    println("labelIncorrectlyUsed: \t"+labelIncorrectlyUsed)
    println("labelNotRecog:        \t"+labelNotRecog)
    println("Label to Color Mapping "+colorlabelMap)
    println("Internal Class Freq:  \t"+classFreqFound)
    println("#EndScore#,%d,%s,%s,%d,%.3f,%.3f,%s,%d,%d,%.3f,%s,%d,%d,%s,%s,%d,%s,%f,%f,%d,%s,%s,%.3f,%.3f,%s,%s".format(
        solverOptions.startTime, solverOptions.runName,solverOptions.gitVersion,(t1MTrain-t0MTrain),solverOptions.dataGenSparsity,solverOptions.dataAddedNoise,if(solverOptions.dataNoiseOnlyTest)"t"else"f",solverOptions.dataGenTrainSize,solverOptions.dataGenCanvasSize,solverOptions.learningRate,if(solverOptions.useMF)"t"else"f",solverOptions.numClasses,MAX_DECODE_ITERATIONS,if(solverOptions.onlyUnary)"t"else"f",if(solverOptions.debug)"t"else"f",solverOptions.roundLimit,if(solverOptions.dataWasGenerated)"t"else"f",avgTestLoss,avgTrainLoss,solverOptions.dataRandSeed , if(solverOptions.useMSRC) "t" else "f", if(solverOptions.useNaiveUnaryMax)"t"else"f" ,avgPerPixTestLoss,avgPerPixTrainLoss, if(solverOptions.trainTestEqual)"t" else "f" , solverOptions.dataSetName ) )
        
  
    sc.stop()

  }

}