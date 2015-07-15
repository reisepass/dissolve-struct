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
import scala.collection.mutable.ListBuffer
 

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
        val binWidth = (maxColor+1)/binsPerColor.asInstanceOf[Double]
        val maxBin = (binsPerColor-1)*binsPerColor*binsPerColor+(binsPerColor-1)*binsPerColor+(binsPerColor-1)+1
        def whichBin(r:Int,g:Int,b:Int):Int={
          val rBin = min((r/binWidth).asInstanceOf[Int],binsPerColor-1)
          val gBin = min((g/binWidth).asInstanceOf[Int],binsPerColor-1)
          val bBin = min((b/binWidth).asInstanceOf[Int],binsPerColor-1)
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
    var countUT = 0
    for(r <- 0 until maxBin; c<- 0 until maxBin){
      if(c>=r){
        linFeat(countUT)=myCoOccur(r)(c)
        countUT+=1
      }
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
    
    
    def colorAverageIntensity1(image:ImageStack,mask:Array[Array[Array[Int]]], maxColorValue:Int=255):Map[Int,Double]={
     val colMod = image.getColorModel()
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
     val supPixMap = new HashMap[Int,Double]
     val supPixMapCount = new HashMap[Int,Int]

          for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
            val lab = mask(x)(y)(z)
            val lastVal = supPixMap.getOrElseUpdate(lab, 0.0)
            val lastCount = supPixMapCount.getOrElseUpdate(lab, 0)
            val col = image.getVoxel(x,y,z).asInstanceOf[Int]
            val r = colMod.getRed(col)
            val g = colMod.getGreen(col)
            val b = colMod.getBlue(col)
            val greyValue = (r/3.0 + g/3.0 + b/3.0)
            supPixMap.put(lab,lastVal+greyValue)
            supPixMapCount.put(lab,lastCount+1)
        }
          val keys = supPixMap.keySet.toList.sorted
          keys.map { key => {
            val totalSum = supPixMap.get(key).get
            val totalCount = supPixMapCount.get(key).get
            (key-> (totalSum/totalCount)/maxColorValue)
          }
          }.toMap 
          
   }
    
    def greyAverageIntensity1(image:ImageStack,mask:Array[Array[Array[Int]]], maxColorValue:Int=255):Map[Int,Double]={
     val colMod = image.getColorModel()
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
     val supPixMap = new HashMap[Int,Double]
     val supPixMapCount = new HashMap[Int,Int]
          
       
          for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
            val lab = mask(x)(y)(z)
            val lastVal = supPixMap.getOrElseUpdate(lab, 0.0)
            val lastCount = supPixMapCount.getOrElseUpdate(lab, 0)
            val col = image.getVoxel(x,y,z).asInstanceOf[Int]
            supPixMap.put(lab,lastVal+col.asInstanceOf[Double])
            supPixMapCount.put(lab,lastCount+1)
        }
          val keys = supPixMap.keySet.toList.sorted
          keys.map { key => {
            val totalSum = supPixMap.get(key).get
            val totalCount = supPixMapCount.get(key).get
            (key-> (totalSum/totalCount)/maxColorValue)
          }
          }.toMap 
          
   }
    
    def greyIntensityVariance(image:ImageStack,mask:Array[Array[Array[Int]]], numSuperPixels:Int,maxColorValue:Int=255 ):Map[Int,Double]={
     val colMod = image.getColorModel()
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
     val  K=  Array.fill(numSuperPixels){-1}
     val  n = Array.fill(numSuperPixels){0}
     val Sum= Array.fill(numSuperPixels){0.0}
     val Sum_sqr = Array.fill(numSuperPixels){0.0}
     
          
       
          for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
            val lab = mask(x)(y)(z)
            val lastK = K(lab)
            val lastn = n(lab)
            val lastSum = Sum(lab)
            val lastSum_sqr = Sum_sqr(lab)
            val grey = image.getVoxel(x,y,z).asInstanceOf[Int]
            if(lastK==(-1))
              K(lab)=grey
            n(lab) = lastn+1
            Sum(lab)=lastSum+ grey-K(lab)
            Sum_sqr(lab)=lastSum_sqr+(grey-K(lab)) *(grey -K(lab))
       
        }
          val variances = for( l<-0 until numSuperPixels) yield{
            ((Sum_sqr(l) - (Sum(l)*Sum(l))/n(l))/n(l))/(maxColorValue*3)//TODO figure out how to properly normalize variance 
          }
          ((0 until numSuperPixels) zip variances).toMap 
   }
    
    
    
    def colorIntensityVariance(image:ImageStack,mask:Array[Array[Array[Int]]], numSuperPixels:Int, maxColorValue:Int=255):Map[Int,Double]={
     val colMod = image.getColorModel()
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
     val  K=  Array.fill(numSuperPixels){-1}
     val  n = Array.fill(numSuperPixels){0}
     val Sum= Array.fill(numSuperPixels){0.0}
     val Sum_sqr = Array.fill(numSuperPixels){0.0}
     
          
       
          for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
            val lab = mask(x)(y)(z)
            val lastK = K(lab)
            val lastn = n(lab)
            val lastSum = Sum(lab)
            val lastSum_sqr = Sum_sqr(lab)
            val col = image.getVoxel(x,y,z).asInstanceOf[Int]
            val grey = colMod.getRed(col)/3 + colMod.getGreen(col)/3 + colMod.getBlue(col)/3
            if(lastK==(-1))
              K(lab)=grey
            n(lab) = lastn+1
            Sum(lab)=lastSum+ grey-K(lab)
            Sum_sqr(lab)=lastSum_sqr+(grey-K(lab)) *(grey -K(lab))
       
        }
          val variances = for( l<-0 until numSuperPixels) yield{
            ((Sum_sqr(l) - (Sum(l)*Sum(l))/n(l))/n(l))/(maxColorValue*3) //TODO figure out how to properly normalize variance 
          }
          ((0 until numSuperPixels) zip variances).toMap 
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
        val maxAbsValue = maxColorValue  
        
        
        val binWidth = (maxAbsValue+1)/histBins.asInstanceOf[Double]
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
               gCoOcMats.get(lab).get(myBin)(neighBin)+=1
               if(myBin!=neighBin)
                 gCoOcMats.get(lab).get(neighBin)(myBin)+=1
               
             })
             
             
           }
   //I only need the upper triagular of the matrix since its symetric. Also i want all my features to be vectorized so
   
   val out=gCoOcMats.map( ((pair:(Int,Array[Array[Int]]))=> {
   val key = pair._1
   val myCoOccur = pair._2
   
        
    val linFeat = Array.fill((histBins * (histBins+1)) / 2){0.0}; 
    var countUT = 0
    for(r <- 0 until histBins; c<- 0 until histBins){
      if(c>=r){
        linFeat(countUT)=myCoOccur(r)(c)
        countUT+=1
      }
    }
    val normFeat = normalize(DenseVector(linFeat.toArray))
 (key,normFeat.toArray)
 })).toMap
        
     out
   }

 
   
   def genMSRCsupPixV2 ( numClasses:Int,S:Int,M:Double ,imageDataSource:String, groundTruthDataSource:String,  featureFn:(ImageStack,Array[Array[Array[Int]]])=>Map[Int,Array[Double]] ,randomSeed:Int =(-1), runName:String = "_", isSquare:Boolean=false,doNotSplit:Boolean=false, debugLabelInFeat:Boolean=false, printMask:Boolean=false, slicNormalizePerClust:Boolean=true, featAddAvgInt:Boolean=false, featAddIntensityVariance:Boolean=false, featAddOffsetColumn:Boolean=false, recompFeat:Boolean=false):(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Map[Int,Int],Map[Int,Double], Array[Array[Double]])={
    
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

         val mPath = if(M==S.asInstanceOf[Double]) "" else "M"+M
         val intFeatPath = if(featAddAvgInt) "t" else ""
         val varFeatPath = if(featAddIntensityVariance) "v" else ""
         val offSetCnstFeatPath = if(featAddOffsetColumn) "of" else ""
         val colorMapPath =  rawImgDir+"/globalColorMap"+".colorLabelMapping2"
         val colorMapF = new File(colorMapPath)
         val colorToLabelMap = if(colorMapF.exists()) GraphUtils.readObjectFromFile[HashMap[Int,Int]](colorMapPath)  else  new HashMap[Int,Int]()
         val classCountPath = rawImgDir+"/classCountMap"+superType+S+"_"+mPath +runName+".classCount"
         val classCountF = new File(classCountPath)
         val oldClassCountFound = classCountF.exists()
         val totalClassCount = if(oldClassCountFound) GraphUtils.readObjectFromFile[HashMap[Int,Double]](classCountPath) else new HashMap[Int,Double]()
     
         val transProbPath = rawImgDir+"/transitionProbabilityMatrix"+superType+S+"_"+mPath +runName+".transProb"
         val transProbF = new File(transProbPath)
         val transProbFound = transProbF.exists()
         val transProb = if(transProbFound) GraphUtils.readObjectFromFile[Array[Array[Double]]](transProbPath) else Array.fill(numClasses,numClasses){0.0}
         var totalConn= if(transProbFound) 1.0 else 0.0
         
        
        assert(allFiles.size>0)
        val allData = for( fI <- 0 until allFiles.size) yield {
        val fName = allFiles(fI)
        val nameNoExt = fName.substring(0,fName.length()-4)
        val rawImagePath =  rawImgDir+"/"+ fName
        
        val graphCachePath = rawImgDir+"/"+ nameNoExt +superType+S+"_"+mPath+intFeatPath+varFeatPath+offSetCnstFeatPath+runName+".graph2"
        val maskpath = rawImgDir+"/"+ nameNoExt +superType+ S+"_"+mPath +runName+".mask"
        val groundCachePath = groundTruthDir+"/"+ nameNoExt+superType+S+"_"+mPath+runName+".ground2"
        val perPixLabelsPath = groundTruthDir+"/"+nameNoExt+superType+S+"_"+mPath+runName+".pxground2"
        val outLabelsPath = groundTruthDir+"/"+ nameNoExt +superType+S+"_"+mPath+intFeatPath+varFeatPath+offSetCnstFeatPath+runName+".labels2"
        
        val cacheMaskF = new File(maskpath)
        val cacheGraphF = new File(graphCachePath)
        val cahceLabelsF =  new File(outLabelsPath)
       
        var imgSrcBidDepth=(-1)
        

        
        if(!recompFeat&&cacheGraphF.exists() && cahceLabelsF.exists()){//Check if its cached 
          
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
          
          
        val allGr =  new SLIC[Int](distFn, rollingAvgFn, normFn, copyImage, S, 15, M,minChangePerIter = 0.002, connectivityOption = "Imperative", debug = false,USE_CLUSTER_MAX_NORMALIZING=slicNormalizePerClust)   
        
        val tMask = System.currentTimeMillis()
        val mask = if( cacheMaskF.exists())  {
          println("But Mask Found")
          GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](maskpath) 
        } else {
          println("and Mask also not Found")
          if(isSquare)
             allGr.calcSimpleSquaresSupPix
          else
            allGr.calcSuperPixels 
          }
        println("Calculate SuperPixels time: "+(System.currentTimeMillis()-tMask))
        if(printMask)
          printSuperPixels(mask,img,300,"_supPix_"+fI+"_"+S+"_"+mPath)//TODO remove me 
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
        println("Now Opening: "+groundTruthpath)
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
       val feat = Array.fill(10){random.nextDouble()}
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
     
      assert(colorToLabelMap.size ==numClasses, " Num Ground Truth Colors Found "+colorToLabelMap.size +" but numClasses set to "+numClasses)
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
     
  
   
   def genMSRCsupPixV3 ( sO:SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] , featureFn:(ImageStack,Array[Array[Array[Int]]],Int,SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels])=>Array[Array[Double]] , afterFeatureFn:(ImageStack,Array[Array[Array[Int]]], IndexedSeq[Node[Vector[Double]]],Int,SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels])=>Array[Node[Vector[Double]]] ):(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Map[Int,Int],Map[Int,Double], Array[Array[Double]])={
    
     
     val numClasses:Int= sO.numClasses
     val S:Int = sO.superPixelSize
     val M:Double= sO.slicCompactness
     val imageDataSource:String= sO.imageDataFilesDir
     val groundTruthDataSource:String=  sO.groundTruthDataFilesDir
     val randomSeed:Int =  sO.dataRandSeed
     val runName:String =  sO.runName
     val isSquare:Boolean= sO.squareSLICoption
     val doNotSplit:Boolean= sO.trainTestEqual 
     val debugLabelInFeat:Boolean=sO.putLabelIntoFeat 
     val  printMask:Boolean=sO.debugPrintSuperPixImg 
     val slicNormalizePerClust:Boolean=sO.slicNormalizePerClust 
     val featAddAvgInt:Boolean=sO.featIncludeMeanIntensity 
     val featAddIntensityVariance:Boolean=sO.featAddIntensityVariance 
     val featAddOffsetColumn:Boolean=sO.featAddOffsetColumn
     val recompFeat:Boolean= sO.recompFeat
    
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

         val mPath = if(M==S.asInstanceOf[Double]) "" else "M"+M
         val intFeatPath = if(featAddAvgInt) "t" else ""
         val varFeatPath = if(featAddIntensityVariance) "v" else ""
         val offSetCnstFeatPath = if(featAddOffsetColumn) "of" else ""
         val colorMapPath =  rawImgDir+"/globalColorMap"+".colorLabelMapping2"
         val colorMapF = new File(colorMapPath)
         val colorToLabelMap = if(colorMapF.exists()) GraphUtils.readObjectFromFile[HashMap[Int,Int]](colorMapPath)  else  new HashMap[Int,Int]()
         val classCountPath = rawImgDir+"/classCountMap"+superType+S+"_"+mPath +runName+".classCount"
         val classCountF = new File(classCountPath)
         val oldClassCountFound = classCountF.exists()
         val totalClassCount = if(oldClassCountFound) GraphUtils.readObjectFromFile[HashMap[Int,Double]](classCountPath) else new HashMap[Int,Double]()
     
         val transProbPath = rawImgDir+"/transitionProbabilityMatrix"+superType+S+"_"+mPath +runName+".transProb"
         val transProbF = new File(transProbPath)
         val transProbFound = transProbF.exists()
         val transProb = if(transProbFound) GraphUtils.readObjectFromFile[Array[Array[Double]]](transProbPath) else Array.fill(numClasses,numClasses){0.0}
         var totalConn= if(transProbFound) 1.0 else 0.0
         
        
        assert(allFiles.size>0)
        val allData = for( fI <- 0 until allFiles.size) yield {
        val fName = allFiles(fI)
        val nameNoExt = fName.substring(0,fName.length()-4)
        val rawImagePath =  rawImgDir+"/"+ fName
        
        val graphCachePath = rawImgDir+"/"+ nameNoExt +superType+S+"_"+mPath+intFeatPath+varFeatPath+offSetCnstFeatPath+runName+".graph2"
        val maskpath = rawImgDir+"/"+ nameNoExt +superType+ S+"_"+mPath +runName+".mask"
        val groundCachePath = groundTruthDir+"/"+ nameNoExt+superType+S+"_"+mPath+runName+".ground2"
        val perPixLabelsPath = groundTruthDir+"/"+nameNoExt+superType+S+"_"+mPath+runName+".pxground2"
        val outLabelsPath = groundTruthDir+"/"+ nameNoExt +superType+S+"_"+mPath+intFeatPath+varFeatPath+offSetCnstFeatPath+runName+".labels2"
        
        val cacheMaskF = new File(maskpath)
        val cacheGraphF = new File(graphCachePath)
        val cahceLabelsF =  new File(outLabelsPath)
       
        var imgSrcBidDepth=(-1)
        

        
        if(!recompFeat&&cacheGraphF.exists() && cahceLabelsF.exists()){//Check if its cached 
          
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
          
          
        val allGr =  new SLIC[Int](distFn, rollingAvgFn, normFn, copyImage, S, 15, M,minChangePerIter = 0.002, connectivityOption = "Imperative", debug = false,USE_CLUSTER_MAX_NORMALIZING=slicNormalizePerClust)   
        
        val tMask = System.currentTimeMillis()
        val mask = if( cacheMaskF.exists())  {
          println("But Mask Found")
          GraphUtils.readObjectFromFile[Array[Array[Array[Int]]]](maskpath) 
        } else {
          println("and Mask also not Found")
          if(isSquare)
             allGr.calcSimpleSquaresSupPix
          else
            allGr.calcSuperPixels 
          }
        println("Calculate SuperPixels time: "+(System.currentTimeMillis()-tMask))
        
        
        if(printMask)
          printSuperPixels(mask,img,300,"_supPix_"+fI+"_"+S+"_"+mPath) 
        if(!cacheMaskF.exists())
         GraphUtils.writeObjectToFile(maskpath,mask)//save a chace 
     
         val tFindCenter = System.currentTimeMillis()
         val (supPixCenter, supPixSize) = allGr.findSupPixelCenterOfMassAndSize(mask)
         println( "Find Center mass time: "+(System.currentTimeMillis()-tFindCenter))
         val tEdges = System.currentTimeMillis()
    val edgeMap = allGr.findEdges_simple(mask, supPixCenter)
    println("Find Graph Connections time: "+(System.currentTimeMillis()-tEdges))

    
    
    
    
    val keys = supPixCenter.keySet.toList.sorted
    assert(keys.equals((0 until keys.size).toList),"The Superpixel id counter skipped something "+keys.mkString(","))
    val numSupPix = keys.size
    val outEdgeMap = edgeMap
    
    
    
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
        println("Now Opening: "+groundTruthpath)
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
    val featureVectors =  featureFn(aStack,mask,numSupPix,sO)
      
      /*
      if(debugLabelInFeat){ //TODO this should be moved into the afterFeatureFn
      
      val tmp = for(id <-0 until keys.size) yield{
       val lab=  groundTruthMap.get(id).get
       val feat = Array.fill(10){random.nextDouble()}
       feat(0)=lab.asInstanceOf[Double]
       (id, feat)
      }
       tmp.toMap
       * 
       */
      
      
   
     
    
   
          println("Compute Features per Blob: "+(System.currentTimeMillis()-tFeat))
    val maxId = max(supPixCenter.keySet)
    val nodesOne = for ( id <- 0 until keys.size) yield{
      Node(id,Vector(featureVectors(id)) , collection.mutable.Set(outEdgeMap.get(id).get.toSeq:_*))
        }
    val outNodes = afterFeatureFn(aStack,mask,nodesOne,numSupPix,sO)
    val outGraph = new GraphStruct[breeze.linalg.Vector[Double],(Int,Int,Int)](Vector(outNodes),maskpath) 
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
     
      assert(colorToLabelMap.size ==numClasses, " Num Ground Truth Colors Found "+colorToLabelMap.size +" but numClasses set to "+numClasses)
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
          
          if(sO.featHistSize>0){
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
      val oo = for ( i<- 0 until numSupPix) yield{ out(i).toArray}
      oo.toArray
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
        
      }
      

      val nor_intens =  normalize(DenseVector(gintens.toArray))
      val nor_neighAvg = normalize(DenseVector(gneighAvg.toArray))
      val nor_neighVar = normalize(DenseVector(gneighVar.toArray))
      val nor_neigh2hAvg = normalize(DenseVector(gneigh2hAvg.toArray))
      val nor_neighh2Var = normalize(DenseVector(gneighh2Var.toArray))
      
      
      val out = nodes.map { old =>  { 
        
        val i = old.idx
        val f = Vector(Array(nor_intens(old.idx)-nor_neighAvg(old.idx),nor_intens(old.idx)/nor_neighAvg(old.idx),nor_neighVar(old.idx))++
            (if(sO.featUnique2Hop) Array((nor_intens(old.idx)-nor_neigh2hAvg(old.idx)),nor_intens(old.idx)/nor_neigh2hAvg(old.idx),nor_neighh2Var(old.idx),
                (nor_neigh2hAvg(old.idx)-nor_neighAvg(old.idx)),nor_neigh2hAvg(old.idx)/nor_neighAvg(old.idx),
                (nor_neighh2Var(old.idx)-nor_neighVar(old.idx)),nor_neighh2Var(old.idx)/nor_neighVar(old.idx)) else Array[Double]() ) 
            ++old.features.toArray)
        new Node[Vector[Double]](i, f,old.connections,nor_intens(i),nor_neighAvg(i),nor_neighVar(i),nor_neigh2hAvg(i),nor_neighh2Var(i))
      }}
      
      out.toArray
    }
    else
      nodes.toArray
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
    sO.roundLimit = options.getOrElse("roundLimit", "5").toInt // After these many passes, each slice of the RDD returns a trained model
    sO.debug = options.getOrElse("debug", "false").toBoolean
    sO.lambda = options.getOrElse("lambda", "0.01").toDouble
    sO.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    sO.doLineSearch = options.getOrElse("linesearch", "true").toBoolean
    sO.debug = options.getOrElse("debug", "false").toBoolean
    sO.onlyUnary = options.getOrElse("onlyUnary", "false").toBoolean
    val MAX_DECODE_ITERATIONS:Int = options.getOrElse("maxDecodeItr",  (if(sO.onlyUnary) 100 else 1000 ).toString ).toInt
    val MAX_DECODE_ITERATIONS_MF_ALT:Int = options.getOrElse("maxDecodeItrMF",  (MAX_DECODE_ITERATIONS).toString ).toInt
    sO.sample = options.getOrElse("sample", "frac")
    sO.sampleFrac = options.getOrElse("samplefrac", "1").toDouble
    sO.dbcfwSeed = options.getOrElse("dbcfwSeed","-1").toInt
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
      assert(sO.useLoopyBP)
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
    sO.dataDepUseIntensityByNeighSD=options.getOrElse("dataDepUseIntensityByNeighSD","false").toBoolean
    sO.dataDepUseIntensityBy2NeighSD=options.getOrElse("dataDepUseIntensityBy2NeighSD","false").toBoolean
    sO.dataDepUseUniqueness=options.getOrElse("dataDepUseUniqueness","false").toBoolean
  
    
    
    
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
       
    
    
   /*
     
   
   val histBinsPerCol = sO.featHistSize/3
   val histBinsPerGray = sO.featHistSize
   val histCoGrayBins = sO.featCoOcurNumBins
   val histCoColorBins = sO.featCoOcurNumBins/3
   
 
   val featFn2 = (image:ImageStack,mask:Array[Array[Array[Int]]])=>{
     val xDim = mask.length
     val yDim = mask(0).length
     val zDim = mask(0)(0).length
     val numSupPix = max(mask.flatten.flatten)+1 //+1 because the id's start at 0 
    
      val bitDep = image.getBitDepth()
        val isColor = if(bitDep==8) false else true //TODO maybe this is not the best way to check for color in the image
        //TODO the bit depth should give me the max value which the hist should span over 
        
        if(isColor){
          
          assert(sO.featHistSize%3==0)
          
          val coreFeat= if(sO.featHistSize>0&&sO.featCoOcurNumBins<=0){
             colorhist(image,mask,histBinsPerCol,255 / histBinsPerCol)
          }
          else if((sO.featHistSize<0&&sO.featCoOcurNumBins>0)){
            coOccurancePerSuperRGB(mask, image, numSupPix, 2)
          }
          else{
          val hist= colorhist(image,mask,histBinsPerCol,255 / histBinsPerCol) 
          val coMat = coOccurancePerSuperRGB(mask, image, numSupPix, 2)
         val combine= hist.map( (a:(Int,Array[Double]))=> { 
            val key = a._1
            val histData = a._2
            (key , histData++coMat.get(key).get)            
          })
          combine
          }
          
          val feat2=if(sO.featIncludeMeanIntensity){
          val avgInts = colorAverageIntensity1(image,mask)
          val out= coreFeat.map( (a:(Int,Array[Double]))=> { 
            val key = a._1
            val pastF = a._2
            (key , Array(avgInts.get(key).get)++pastF)            
          })
          out
          }
          else{
            coreFeat
          }
          
          val feat3=if(sO.featAddIntensityVariance){
           val newFeat = colorIntensityVariance(image,mask,numSupPix)
          val out= feat2.map( (a:(Int,Array[Double]))=> { 
            val key = a._1
            val pastF = a._2
            (key , pastF++Array(newFeat.get(key).get))            
          })
          out
         }
         else{
           feat2
         }
         
         if(sO.featAddOffsetColumn){
           val newFeat =  ((0 until numSupPix) zip List.fill(numSupPix){1.0}).toMap
          val out= feat3.map( (a:(Int,Array[Double]))=> { 
            val key = a._1
            val pastF = a._2
            (key , pastF++Array(newFeat.get(key).get))            
          })
          out
         }
         else{
           feat3
         }
          
          
        }
       else{
        
         val coreFeat=if(sO.featHistSize>0&&sO.featCoOcurNumBins<=0){
            greyHist(image,mask,histBinsPerGray,255 / (histBinsPerGray))
          
         }
         else if((sO.featHistSize<0&&sO.featCoOcurNumBins>0)){
           greyCoOccurancePerSuper(image, mask, histCoGrayBins)
         }
         else{
          val hist=greyHist(image,mask,histBinsPerGray,255 / (histBinsPerGray))
          val coMat= greyCoOccurancePerSuper(image, mask, histCoGrayBins)
          val combine= hist.map( (a:(Int,Array[Double]))=> { 
            val key = a._1
            val histData = a._2
            (key , histData++coMat.get(key).get)            
          })
          combine
         }
         
         val feat2=if(sO.featIncludeMeanIntensity){
          val avgInts = greyAverageIntensity1(image,mask)
          val out= coreFeat.map( (a:(Int,Array[Double]))=> { 
            val key = a._1
            val pastF = a._2
            (key , Array(avgInts.get(key).get)++pastF)            
          })
          out
         }
         else{
           coreFeat
         }
         
         val feat3=if(sO.featAddIntensityVariance){
           val newFeat = greyIntensityVariance(image,mask,numSupPix)
          val out= feat2.map( (a:(Int,Array[Double]))=> { 
            val key = a._1
            val pastF = a._2
            (key , pastF++Array(newFeat.get(key).get))            
          })
          out
         }
         else{
           feat2
         }
         
         if(sO.featAddOffsetColumn){
           val newFeat =  ((0 until numSupPix) zip List.fill(numSupPix){1.0}).toMap
          val out= feat3.map( (a:(Int,Array[Double]))=> { 
            val key = a._1
            val pastF = a._2
            (key , pastF++Array(newFeat.get(key).get))            
          })
          out
         }
         else{
           feat3
         }
          
       }      
   }
  // 
    
   
   
 
    
    //TODO add features to this noise creator which makes groundTruth files just like those in getMSRC or getMSRCSupPix
   val (ootrainData,ootestData, oocolorlabelMap, ooclassFreqFound,ootransProb) = genMSRCsupPixV2(sO.numClasses, sO.superPixelSize, sO.slicCompactness,sO.imageDataFilesDir, sO.groundTruthDataFilesDir, featFn2, sO.dataRandSeed, sO.runName, sO.squareSLICoption, sO.trainTestEqual,sO.putLabelIntoFeat,sO.debugPrintSuperPixImg,sO.slicNormalizePerClust,sO.featIncludeMeanIntensity,sO.featAddIntensityVariance,sO.featAddOffsetColumn,sO.recompFeat) 
  */
   val (trainData,testData, colorlabelMap, classFreqFound,transProb) = genMSRCsupPixV3(sO,featFn3,afterFeatFn1)
    
    
   println("Train SgreyHistize:"+trainData.size)
    println("Test Size:"+testData.size)
    
    

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
    
    if(sO.useMSRC) sO.numClasses = 24


    println(sO.toString())

    val conf =
      if (runLocally)
        new SparkConf().setAppName(appName).setMaster("local")
      else
        new SparkConf().setAppName(appName)

    val sc = new SparkContext(conf)
    sc.setCheckpointDir(debugDir + "/checkpoint-files")

    println(SolverUtils.getSparkConfString(sc.getConf))

    
    
    
    var numGraidBins = 2
    val graidientFunc= if(sO.dataDepUseIntensity) { numGraidBins = 2; (a:Node[Vector[Double]],b:Node[Vector[Double]])=>{ if(Math.abs(a.avgValue-b.avgValue)<0.2) 0 else 1} }
    else if (sO.dataDepUseIntensityBy2NeighSD) {numGraidBins = 3;  (a:Node[Vector[Double]],b:Node[Vector[Double]])=>{ 
      
      val sd= Math.sqrt(a.hop2NeighVar)
      val dif = (Math.abs(a.avgValue-b.avgValue)/sd)*20
      
      max(0,min(dif.asInstanceOf[Int],numGraidBins-1))
     }  
    
    } 
    else if(sO.dataDepUseIntensityByNeighSD){
      numGraidBins = 3;  (a:Node[Vector[Double]],b:Node[Vector[Double]])=>{ 
      
      val sd= Math.sqrt(a.neighVariance)
      val dif = (Math.abs(a.avgValue-b.avgValue)/sd)*20
      
      max(0,min(dif.asInstanceOf[Int],numGraidBins-1))
     }  
    }
    else{ // if ( dataDepUseUniqueness )
      numGraidBins = 3;  (a:Node[Vector[Double]],b:Node[Vector[Double]])=>{ 
      
      val sdA= Math.sqrt(a.neighVariance)
      val uniqA = Math.abs(a.avgValue-a.neighMean)/sdA
      val sdB= Math.sqrt(b.neighVariance)
      val uniqB = Math.abs(b.avgValue-b.neighMean)/sdB
      
      if(uniqA>1 && uniqB>1) 0 else if((uniqA>1 && uniqB<1 )||(uniqA<1 && uniqB>1 )) 1 else 2
     }  
    }
    
    
    sO.testDataRDD =
      if (sO.enableManualPartitionSize)
        Some(sc.parallelize(trainData, sO.NUM_PART))
      else
        Some(sc.parallelize(trainData))

    val trainDataRDD =
      if (sO.enableManualPartitionSize)
        sc.parallelize(trainData, sO.NUM_PART)
      else
        sc.parallelize(trainData)

        val myGraphSegObj = if(!sO.modelPairwiseDataDependent) {new GraphSegmentationClass(sO.onlyUnary,MAX_DECODE_ITERATIONS,
            sO.learningRate ,sO.useMF,sO.mfTemp,sO.useNaiveUnaryMax,
            DEBUG_COMPARE_MF_FACTORIE,MAX_DECODE_ITERATIONS_MF_ALT,sO.runName,
            if(sO.useClassFreqWeighting) classFreqFound else null,
            sO.weighDownUnary,sO.weighDownPairwise, sO.LOSS_AUGMENTATION_OVERRIDE,
            false,sO.PAIRWISE_UPPER_TRI,sO.useMPLP,sO.useLoopyBP) }
        else {
          new GraphSegDataDepPair(graidientFunc,numGraidBins,sO.runName,if(sO.useClassFreqWeighting)classFreqFound else null)
        }
    val trainer: StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] =
      new StructSVMWithDBCFW[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels](
        trainDataRDD,
        myGraphSegObj,
        sO)

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
    println("\nTest Avg Loss : Sup(" + avgTestLoss +") PerPix("+avgPerPixTestLoss+ ") numItems " + testData.size)
    println()
    println("Occurances:           \t"+labelOccurance)
    println("labelIncorrectlyUsed: \t"+labelIncorrectlyUsed)
    println("labelNotRecog:        \t"+labelNotRecog)
    println("Label to Color Mapping "+colorlabelMap)
    println("Internal Class Freq:  \t"+classFreqFound)
    
     def bToS( a:Boolean)={if(a)"t"else"f"}
       
    
    val newLabels = sO.sampleFrac+","+ (if(sO.doWeightedAveraging) "t" else "f")+","+ 
            (if(sO.onlyUnary) "t" else "f") +","+(if(sO.squareSLICoption) "t" else "f")+","+ sO.superPixelSize+","+ sO.dataSetName+","+( if(sO.trainTestEqual)"t" else "f")+","+
            sO.inferenceMethod+","+sO.dbcfwSeed+","+ (if(sO.dataGenGreyOnly) "t" else "f")+","+ (if(sO.compPerPixLoss) "t" else "f")+","+ sO.dataGenNeighProb+","+ sO.featHistSize+","+
            sO.featCoOcurNumBins+","+ (if(sO.useLoopyBP) "t" else "f")+","+ (if(sO.useMPLP) "t" else "f")+","+ (if(sO.slicNormalizePerClust) "t" else "f")+","+ sO.dataGenOsilNoise+","+ sO.dataRandSeed+","+
            sO.dataGenHowMany+","+sO.slicCompactness+","+( if(sO.putLabelIntoFeat) "t" else "f" )+","+sO.dataAddedNoise+","+(if(sO.modelPairwiseDataDependent) "t" else "f")+","+(if(sO.featIncludeMeanIntensity) "t" else "f")+
            bToS(sO.featAddOffsetColumn)+","+bToS(sO.featAddIntensityVariance)
    
    
    
    println("#EndScore#,%d,%s,%s,%d,%.3f,%.3f,%s,%d,%d,%.3f,%s,%d,%d,%s,%s,%d,%s,%f,%f,%d,%s,%s,%.3f,%.3f,%s,%s".format(
        sO.startTime, sO.runName,sO.gitVersion,(t1MTrain-t0MTrain),sO.dataGenSparsity,sO.dataAddedNoise,if(sO.dataNoiseOnlyTest)"t"else"f",sO.dataGenTrainSize,
        sO.dataGenCanvasSize,sO.learningRate,if(sO.useMF)"t"else"f",sO.numClasses,MAX_DECODE_ITERATIONS,if(sO.onlyUnary)"t"else"f",
        if(sO.debug)"t"else"f",sO.roundLimit,if(sO.dataWasGenerated)"t"else"f",avgTestLoss,avgTrainLoss,sO.dataRandSeed , 
        if(sO.useMSRC) "t" else "f", if(sO.useNaiveUnaryMax)"t"else"f" ,avgPerPixTestLoss,avgPerPixTrainLoss, if(sO.trainTestEqual)"t" else "f" , 
        sO.dataSetName )+newLabels )
        
  
    sc.stop()

  }

}