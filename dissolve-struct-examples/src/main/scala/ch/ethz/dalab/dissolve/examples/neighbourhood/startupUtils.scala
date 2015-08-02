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
import ij.process.ImageProcessor
import ij.process.ByteProcessor
import ij.ImageStack
import ij.IJ
import ij.ImagePlus
import scala.sys.process._



/**
 * @author mort


 */



object startupUtils {
  
    
  def splitLargeTiffStack(dataDir:String, howOften:Int){
    splitLargeTiffStack(dataDir+"/Images",dataDir+"/GroundTruth",howOften)
  }
    def splitLargeTiffStack(rawImgDir:String,groundTruthDir:String, howOften:Int){
      
     val oldTifs =  Option(new File(rawImgDir).list).map(_.filter(_.endsWith(".tif"))).get
     assert(oldTifs!=null&&oldTifs.size>0, "Did not find any .tif images in this directory. Rename tiff to tif if needed")
  
     val workingDir = new java.io.File(rawImgDir.trim+"/..")
     val backM = sys.process.Process(Seq("mkdir","backup"),workingDir)
     if((backM.!)!=0)
     { println("#WARNING# since the backup folder alreayd excists we are assuming these files have alright been properly split and will do nothing additionaly to the images");
     return; 
     }
     sys.process.Process(Seq("mkdir","backup/Images"),workingDir).!!     
     sys.process.Process(Seq("mkdir","backup/GroundTruth"),workingDir).!!
     sys.process.Process(Seq("bash","-c","mv Images/* backup/Images/"),workingDir).!!
     sys.process.Process(Seq("bash","-c","mv GroundTruth/* backup/GroundTruth/"),workingDir).!!
     sys.process.Process(Seq("bash","-c","mv *.cfg backup/"),workingDir).!!
     
   val lotsofTIFF =  Option(new File(rawImgDir+"/../backup/Images").list).map(_.filter(_.endsWith(".tif"))).get
     assert(lotsofTIFF!=null&&lotsofTIFF.size>0, "Did not find any .tif images in this directory. Rename tiff to tif if needed")
  
     
     for ( itr <- 0 until lotsofTIFF.size){
       
        val fileName = lotsofTIFF(itr)
        val rawImagePath = rawImgDir+"/../backup/Images/"+fileName
        val nameNoExt = fileName.substring(0,fileName.length()-4)
        
        val tStartPre = System.currentTimeMillis()
        val opener = new Opener();
        val img = opener.openImage(rawImagePath);
        val rawStack = img.getStack
        val bitDep = rawStack.getBitDepth()
        val colMod = rawStack.getColorModel()
        val xDim = rawStack.getWidth
        val yDim = rawStack.getHeight
        val zDim = rawStack.getSize
        val groundTruthpath =  groundTruthDir+"/../backup/GroundTruth/"+ nameNoExt+"_GT"+".tif"
        val openerGT = new Opener();
        val imgGT = openerGT.openImage(groundTruthpath);
        
        val gtStack = imgGT.getStack
        assert(xDim==gtStack.getWidth)
        assert(yDim==gtStack.getHeight)
        assert(zDim==gtStack.getSize)
        val minDim = min(min(xDim,yDim),zDim)
        val minCubSize = minDim/howOften
        val subXCount = xDim/minCubSize
        val subYCount = yDim/minCubSize
        val subZCount = zDim/minCubSize
        val remainderX = xDim%minCubSize
        val remainederY = yDim%minCubSize
        val remainederZ = zDim%minCubSize
        
        
        var counter =0;
        
        val subImgRaw = Array.fill(subXCount){Array.fill(subYCount){Array.fill(subZCount){ new ImageStack()}}}
        val subImgGT = Array.fill(subXCount){Array.fill(subYCount){Array.fill(subZCount){ new ImageStack()}}}
          for ( subX <- 0 until subXCount; subY <- 0 until subYCount;subZ <- 0 until subZCount){  
            val title = nameNoExt+"_subImg"+counter+".tif"
            val titleGT = nameNoExt+"_subImg"+counter+"_GT.tif"
            counter+=1;
            val xSubDim =  (if(subX==(subXCount-1)) remainderX+minCubSize else minCubSize)
            val ySubDim =  (if(subY==(subYCount-1)) remainederY+minCubSize else minCubSize)
            val zSubDim = (if(subZ==subZCount-1) remainederZ+minCubSize else minCubSize)
            val subimp = IJ.createImage(title," "+bitDep,xSubDim, ySubDim, zSubDim)
            val subimpGT = IJ.createImage(titleGT," "+bitDep,xSubDim, ySubDim, zSubDim)
            subImgRaw(subX)(subY)(subZ)=subimp.getStack
            subImgGT(subX)(subY)(subZ)=subimpGT.getStack
        }
        
        for ( x<- 0 until xDim ; y<- 0 until yDim; z<- 0 until zDim){
          val subX = min(x/minCubSize,subXCount-1)
          val subY = min(y/minCubSize,subYCount-1)
          val subZ = min(z/minCubSize,subZCount-1)
          val offSetX = subX*minCubSize
          val offSetY = subY*minCubSize
          val offSetZ = subZ*minCubSize
          val localX = x-offSetX
          val localY = y-offSetY
          val localZ = z-offSetZ
          subImgRaw(subX)(subY)(subZ).setVoxel(localX,localY,localZ,rawStack.getVoxel(x,y,z))
          subImgGT(subX)(subY)(subZ).setVoxel(localX,localY,localZ,gtStack.getVoxel(x,y,z))
        }
        
         for ( subX <- 0 until subXCount; subY <- 0 until subYCount;subZ <- 0 until subZCount){  
            val title = nameNoExt+"_subImg"+subX+"-"+subY+"-"+subZ+".tif"
            val titleGT = nameNoExt+"_subImg"+subX+"-"+subY+"-"+subZ+"_GT.tif"
            
            val xSubDim =  (if(subX==(subXCount-1)) remainderX+minCubSize else minCubSize)
            val ySubDim =  (if(subY==(subYCount-1)) remainederY+minCubSize else minCubSize)
            val zSubDim = (if(subZ==subZCount-1) remainederZ+minCubSize else minCubSize)
            val subimp = IJ.createImage(title," "+bitDep,xSubDim, ySubDim, zSubDim)
            val subimpgt = IJ.createImage(titleGT," "+bitDep,xSubDim, ySubDim, zSubDim)
            subimp.setStack(subImgRaw(subX)(subY)(subZ))
            subimpgt.setStack(subImgGT(subX)(subY)(subZ))
          IJ.saveAs(subimp, "tif", rawImgDir+"/" + title); 
          IJ.saveAs(subimpgt, "tif", groundTruthDir+"/" + titleGT);
         }
       
       
     }
    }
    
  def printMemory() {
    val mb = 1024*1024;
         
        //Getting the runtime reference from system
        val runtime = Runtime.getRuntime();
         
        System.out.println("##### Heap utilization statistics [MB] #####");
         
        //Print used memory
        System.out.println("Used Memory:"
            + (runtime.totalMemory() - runtime.freeMemory()) / mb);
 
        //Print free memory
        System.out.println("Free Memory:"
            + runtime.freeMemory() / mb);
         
        //Print total available memory
        System.out.println("Total Memory:" + runtime.totalMemory() / mb);
 
        //Print Maximum available memory
        System.out.println("Max Memory:" + runtime.maxMemory() / mb);

        
        
  }
  
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
   def greyHistv2  (image:ImageStack,mask:Array[Array[Array[Int]]],numSupPix:Int,sO:SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]):Array[Array[Double]]={
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
        val histBinsPerCol = if(sO.isColor) sO.featHistSize/3 else sO.featHistSize
        val histWidt = sO.maxColorValue/histBinsPerCol
        
     val hists = Array.fill(numSupPix){Array.fill(sO.featHistSize){0.0}}
        
         for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
            val lab = mask(x)(y)(z)
            
            val col = image.getVoxel(x,y,z).asInstanceOf[Int]
        
            hists(lab)(min(histBinsPerCol - 1, (col / histWidt))) += 1.0
        }
        val out = for( i <- 0 until numSupPix) yield {
          normalize(DenseVector(hists(i))).toArray
        }
        out.toArray
        
   }
   
   //TODO Remove this function:  It is bad because it relies on the global SD and global Mean. But if this were actually relevelant then it would be different for Training and Test. And hence what you train on will be shifted differently so it means something different in test data
   def greyHistvStd  (image:ImageStack,mask:Array[Array[Array[Int]]],numSupPix:Int,sO:SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]):Array[Array[Double]]={
        val xDim = image.getWidth
        val yDim = image.getHeight
        val zDim = image.getSize
        
        
        
     val hists = Array.fill(numSupPix){Array.fill(sO.featHistSize*2){0.0}}
        val globalSD = sqrt(sO.globalVar)
        val maxAbsDif = globalSD*2
        val binWidth = maxAbsDif/sO.featHistSize
         for( x<- 0 until xDim; y<-0 until yDim ; z <- 0 until zDim){
            val lab = mask(x)(y)(z)
            
            val col = image.getVoxel(x,y,z).asInstanceOf[Int].asInstanceOf[Double]
            val difMean = col - sO.globalMean
            val binSigned = Math.abs((difMean/binWidth)).asInstanceOf[Int]
            if(difMean<0)
              hists(lab)(min(sO.featHistSize-1,binSigned) ) += 1.0
            else
              hists(lab)(min(sO.featHistSize*2-1,binSigned+sO.featHistSize) ) += 1.0
            
        }
        val out = for( i <- 0 until numSupPix) yield {
          normalize(DenseVector(hists(i))).toArray
        }
        out.toArray
        
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

 
   
   
   def rowMajor3Dto1D (x:Int,y:Int,z:Int,xDim:Int,yDim:Int,zDim:Int):Int={
    x*(zDim*yDim)+y*(zDim)+z
  }
   
   def columnMajor3Dto1D (x:Int,y:Int,z:Int,xDim:Int,yDim:Int,zDim:Int):Int={
    x+y*(xDim)+z*(xDim*yDim)  
   }
   
   def imgStackToArray (aStack:ImageStack):Array[Double]={
     val xDim = aStack.getWidth
        val yDim = aStack.getHeight
        val zDim = aStack.getSize
        val copyImage = Array.fill(xDim*yDim*zDim) {0.0}
        for (x <- 0 until xDim; y <- 0 until yDim; z <- 0 until zDim) {
          copyImage(columnMajor3Dto1D(x,y,z,xDim,yDim,zDim))= aStack.getVoxel(x, y, z)
        }
        copyImage
   }
   
   def quantileDataDepFn( dataDepValue:((Node[Vector[Double]],Node[Vector[Double]])=>Double),dataDepNumBins:Int, allData:  Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]]):Array[Double]={
     
     val allDatDepVal=allData.flatMap( lblObj => {
       lblObj.pattern.graphNodes.toArray.flatMap( curNode => {
         curNode.connections.map {  neighId => 
           dataDepValue(curNode,lblObj.pattern.get(neighId)) }
       })
       })
       
     val sortedVal=allDatDepVal.sorted
     val bucketSize = (Math.floor(sortedVal.size/dataDepNumBins)).asInstanceOf[Int]
     
     val bounds = (1 until dataDepNumBins).toList.map { boundI => sortedVal(boundI*bucketSize) }.toArray
     return(bounds ++ Array(Double.MaxValue))
   }
   
   def colImgStackToGreyArray (aStack:ImageStack):Array[Double]={
     val xDim = aStack.getWidth
        val yDim = aStack.getHeight
        val zDim = aStack.getSize
        val cMod = aStack.getColorModel
        val copyImage = Array.fill(xDim*yDim*zDim) {0.0}
        for (x <- 0 until xDim; y <- 0 until yDim; z <- 0 until zDim) {
           val r = cMod.getRed(aStack.getVoxel(x, y, z).asInstanceOf[Int])
           val g = cMod.getGreen(aStack.getVoxel(x, y, z).asInstanceOf[Int])
           val b = cMod.getBlue(aStack.getVoxel(x, y, z).asInstanceOf[Int])
          copyImage(columnMajor3Dto1D(x,y,z,xDim,yDim,zDim))= (r/3 + g/3 + b/3)
        }
        copyImage
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
        if(printMask)
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
   
   def genMSRCsupPixV3 ( sO:SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] , featureFn:(ImageStack,Array[Array[Array[Int]]],Int,SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels])=>Array[Array[Double]] , afterFeatureFn:(ImageStack,Array[Array[Array[Int]]], IndexedSeq[Node[Vector[Double]]],Int,SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels])=>Array[Node[Vector[Double]]] ):(Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Seq[LabeledObject[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels]],Map[Int,Int],Map[Int,Double], Array[Array[Double]],SolverOptions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] )={
    
     
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
         val intFeatPath = "" //if(featAddAvgInt) "t" else "" //I disabled this because the feature options are really complicated so i should just recompute it every time
         val varFeatPath = "" //if(featAddIntensityVariance) "v" else ""  //I disabled this because the feature options are really complicated so i should just recompute it every time
         val offSetCnstFeatPath = "" // if(featAddOffsetColumn) "of" else ""  //I disabled this because the feature options are really complicated so i should just recompute it every time
         val minBlobSizePath = if(sO.slicMinBlobSize>=0) "mB"+sO.slicMinBlobSize else ""
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
         
         
         
        val (globalMeanIntensity, globalVariance, globalMin, globalMax) = if(sO.preStandardizeImagesFirst){
         
       
        
          
          var gMin = Double.MaxValue
          var gMax = Double.MinValue
          val tStartPre = System.currentTimeMillis()
          val aggrStat= for( fI <- 0 until allFiles.size) yield {
          val fName = allFiles(fI)
          val nameNoExt = fName.substring(0,fName.length()-4)
          val rawImagePath =  rawImgDir+"/"+ fName
          val opener = new Opener();
          val img = opener.openImage(rawImagePath);
          val aStack = img.getStack
          val bitDep = aStack.getBitDepth()
          val isColor = if(bitDep==8) false else true  
          assert(sO.isColor==isColor,"Looks like your image is not the color model you specified")
          assert(sO.isColor==false,"image standardizeation is not yet implemented for color")
          val xDim = aStack.getWidth
          val yDim = aStack.getHeight
          val zDim = aStack.getSize
          val copyImage =  DenseVector(imgStackToArray(aStack))
          val minOut = min(copyImage)
          if(minOut<gMin)
            gMin=minOut
          val maxOut = max(copyImage)
          if(maxOut>gMax)
            gMax = minOut
          val sumOut = sum(copyImage)
          val countOut = copyImage.size
          val meanOut = sumOut/countOut
          val varOut = variance(copyImage)
          (countOut,meanOut,varOut)
        }
        var movingVar = 0.0
        var movingMean = 0.0
        var movingCount = 0
        for( i <- 0 until aggrStat.size){
          movingVar= (movingVar*movingCount + aggrStat(i)._3*aggrStat(i)._1)/(movingCount+aggrStat(i)._1)
          movingMean = (movingMean*movingCount + aggrStat(i)._2*aggrStat(i)._1)/(movingCount+aggrStat(i)._1)
          movingCount += aggrStat(i)._1
        }
        println("Time to comp global stats: "+(System.currentTimeMillis()-tStartPre))
          (movingMean,movingVar,gMin,gMax)
        }
        else{
          (0.0,1.0,0,sO.maxColorValue)
        }
        sO.globalMean=globalMeanIntensity
        sO.globalVar =globalVariance
         
         
        
        val allData = for( fI <- 0 until allFiles.size) yield {
        val fName = allFiles(fI)
        val nameNoExt = fName.substring(0,fName.length()-4)
        val rawImagePath =  rawImgDir+"/"+ fName
        
        val graphCachePath = rawImgDir+"/"+ nameNoExt +superType+S+"_"+mPath+minBlobSizePath+intFeatPath+varFeatPath+offSetCnstFeatPath+runName+".graph2"
        val maskpath = rawImgDir+"/"+ nameNoExt +superType+ S+"_"+mPath+minBlobSizePath +runName+".mask"
        val groundCachePath = groundTruthDir+"/"+ nameNoExt+superType+S+"_"+mPath+minBlobSizePath+runName+".ground2"
        val perPixLabelsPath = groundTruthDir+"/"+nameNoExt+superType+S+"_"+mPath+minBlobSizePath+runName+".pxground2"
        val outLabelsPath = groundTruthDir+"/"+ nameNoExt +superType+S+"_"+mPath+minBlobSizePath+intFeatPath+varFeatPath+offSetCnstFeatPath+runName+".labels2"
        
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
          
          
        val allGr =  new SLIC[Int](distFn, rollingAvgFn, normFn, copyImage, S, 15, M,minChangePerIter = 0.002, connectivityOption = "Imperative", debug = false,USE_CLUSTER_MAX_NORMALIZING=slicNormalizePerClust, in_minBlobSize = sO.slicMinBlobSize )   
        
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
        
        
        
        if(!cacheMaskF.exists())
         GraphUtils.writeObjectToFile(maskpath,mask)//save a chace 
     
         val tFindCenter = System.currentTimeMillis()
         val (supPixCenter, supPixSize) = allGr.findSupPixelCenterOfMassAndSize(mask)
         println( "Find Center mass time: "+(System.currentTimeMillis()-tFindCenter))
         val tEdges = System.currentTimeMillis()
  
     //    val edgeMap = allGr.findEdges_simple(mask, supPixCenter)
        val keys = supPixCenter.keySet.toList.sorted
    assert(keys.equals((0 until keys.size).toList),"The Superpixel id counter skipped something "+keys.mkString(","))
    println("Find Graph Connections time: "+(System.currentTimeMillis()-tEdges))

    
    
    
    

    val numSupPix = keys.size
    val tEdges2 = System.currentTimeMillis()
    val edgeMap2 = allGr.findEdges_trueIter(mask,numSupPix)
      println("Find Graph Connections time NEW: "+(System.currentTimeMillis()-tEdges2))

    val outEdgeMap = edgeMap2 //edgeMap
    
    
    
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
      
      
      if(printMask){
          printSuperPixels(mask,img,300,"_supPix_"+nameNoExt+"_"+fI+"_"+S+"_"+mPath+minBlobSizePath)
          printSuperPixels(mask,imgGT,110,"_supPix_"+nameNoExt+"_"+fI+"_"+S+"_"+mPath+minBlobSizePath+"_GT") 
          
        }
   
     
    
   
          println("Compute Features per Blob: "+(System.currentTimeMillis()-tFeat))
    val maxId = max(supPixCenter.keySet)
    val nodesOne = for ( id <- 0 until keys.size) yield{
      Node(id,Vector(featureVectors(id)) , collection.mutable.Set(outEdgeMap(id).toSeq:_*))
        }
    val outNodes = afterFeatureFn(aStack,mask,nodesOne,numSupPix,sO)
    val outGraph = new GraphStruct[breeze.linalg.Vector[Double],(Int,Int,Int)](Vector(outNodes),maskpath) 
     GraphUtils.writeObjectToFile(graphCachePath,outGraph)    
  
         
         
        val labelsInOrder = (0 until groundTruthMap.size).map(a=>groundTruthMap.get(a).get)
        val outLabels = new GraphLabels(Vector(labelsInOrder.toArray),numClasses,groundTruthpath)
         GraphUtils.writeObjectToFile(outLabelsPath,outLabels)
         
              if(!transProbFound){
            
            for ( i <- 0 until numSupPix){
              val leftN = i
              val leftClass = labelsInOrder(leftN)
              val neigh = outEdgeMap(i)
             
              neigh.foreach { thisNeigh => {
                val rightClass = labelsInOrder(thisNeigh)
                transProb(leftClass)(rightClass)+=1.0
                transProb(rightClass)(leftClass)+=1.0
                totalConn+=2.0
              } }
            }
            
          }
         
         //TODO remove debugging 
    if(printMask)
        GraphUtils.printSupPixLabels_Int(outGraph,outLabels,0,nameNoExt+"_"+fI+"_"+"_idMap_",colorMap=colorToLabelMap.toMap.map(_.swap))
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
      return (training,test,colorToLabelMap.toMap,classFreq.toMap,outTransProb,sO)
  }
     
  
}