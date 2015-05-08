package ch.ethz.dalab.dissolve.examples.neighbourhood

import java.io.File
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import breeze.linalg.{ Matrix, Vector }
import ch.ethz.dalab.dissolve.regression.LabeledObject
import scala.io.Source
import breeze.linalg.DenseMatrix
import java.awt.image.DataBufferInt
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.max
import breeze.linalg.min
import breeze.linalg.normalize
import breeze.math._
import breeze.numerics._
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import java.io.PrintWriter
import java.awt.image.DataBufferInt
import ch.ethz.dalab.dissolve.examples.imageseg
import org.apache.log4j.PropertyConfigurator
import org.apache.spark.sql.catalyst.expressions.IsNull
import scala.collection.mutable.HashMap
import scala.util.Random
import scala.collection.mutable.HashSet
import breeze.stats.DescriptiveStats._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.LinkedList
import ch.ethz.dalab.dissolve.examples.imageseg.ROIFeature
import ch.ethz.dalab.dissolve.examples.imageseg.ROILabel
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationDemo
import java.awt.Graphics2D
import scala.collection.mutable.ArrayBuffer
import java.awt.geom.AffineTransform
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationUtils



class GraphStruct[Features, OriginalCoord](graph: Vector[Node[Features]],
                                           dataLink: HashMap[Int, OriginalCoord],
                                           maxCoordnate: OriginalCoord) extends Serializable{ //dataLink could be a Vector, no reason for hashMap

  def graphNodes = graph
  def dataGraphLink = dataLink
  def maxCoord = maxCoordnate
//TODO Add?? mapping from original xyz to nodes. For prediction we need something like this but it could be more efficient to just have a print out method that returns a xyz matrix 
  def getF(i:Int): Features ={graphNodes(i).features}
  def get(i:Int): Node[Features] = { graphNodes(i)}
  def getC(i:Int): scala.collection.mutable.Set[Int] = { graphNodes(i).connections}
  def size = graphNodes.size
   
}

case class Node[Features](
  val idx: Int,
  val features: Features,
  val connections: scala.collection.mutable.Set[Int]) extends Serializable {
  override def hashCode() : Int={
    idx
  }


}


case class GraphLabels ( d:Vector[Int],numClasses:Int)extends Serializable {
  assert(numClasses>0)
  def isInverseOf(other: GraphLabels): Boolean = {
    if(other.d.size !=d.size)
      return false
    else{
      for( i <- 0 until d.size){
        if (other.d(i)==d(i))
          return false
      }
      return true
    }
  }
}



object GraphUtils {
  
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
  
  type D3ArrInt = Array[Array[Array[Int]]]
  type D3ArrDbl = Array[Array[Array[Double]]]
      def randomVec () : Array[Double]={
      Array.fill[Double](10){Math.random()}
     
    }
  
  
  def reConstruct3dMat(labels:GraphLabels, dataLink: HashMap[Int,(Int,Int,Int)],xDim:Int,yDim:Int,zDim:Int): D3ArrInt= {
    
    val out = Array.fill(xDim,yDim,zDim){0}
    for( i<-0 until labels.d.size){
      val coords = dataLink.get(i).get
      out(coords._1)(coords._2)(coords._3)=labels.d(i)
    }
    out
  }
  
  //TODO find a nice scala-esq way of doing this. essencially i want the matlab function 'squeeze'
  def flatten3rdDim(in:D3ArrInt):Array[Array[Int]]={ 
      
    val xDim = in.length
    val yDim = in(0).length
    val out = Array.fill(xDim,yDim){0}
    for( x <-0 until in.length){
        for(y <-0 until in(0).length){
          out(x)(y)=in(x)(y)(0)  
        }
      }
    return out

  }
  def printBMPfrom3dMat (in:Array[Array[Int]], fileName:String)={
   
    val xDim = in.length
    val yDim = in(0).length
    val img: BufferedImage  = new BufferedImage(xDim, yDim,
    BufferedImage.TYPE_INT_RGB);
    for( x <- 0 until xDim){
      for (y <- 0 until yDim){
        val tmp =ImageSegmentationUtils.colormapRev
        val toGet = in(x)(y)
        val tmpRGB = ImageSegmentationUtils.colormapRev.get(toGet)
        if ( tmpRGB.isEmpty){
          print("oops")
        }
        val rgb = tmpRGB.get
        img.setRGB(x, y, rgb)
      }
    }
    
    //TODO change the path. We dont want to assume things about the file structure below the pwd 
    ImageIO.write(img, "BMP", new File("../data/debug/"+fileName));
    
  }
  
  

  
  
  def nodeCmp(e1: Node[_], e2: Node[_]) = (e1.idx <   e2.idx)

def d3randomVecDlb () : D3ArrDbl={
      Array.fill[Double](10,10,10){Math.random()}
     
    }
def d3randomVecInt (x:Int =10, y:Int = 10, z:Int= 10) : D3ArrInt={
      Array.fill(x,y,z){(Math.random()*255).asInstanceOf[Int]}
    }


def convertOT_msrc_toGraph ( xi: DenseMatrix[ROIFeature], yi: DenseMatrix[ROILabel], numClasses : Int ): (GraphStruct[Vector[Double], (Int,Int,Int)], GraphLabels) ={
  
  
  val nodeList = new scala.collection.mutable.ListBuffer[Node[Vector[Double]]]
  val labelVect = Array.fill(xi.rows*xi.cols)(0)
   val linkCoord = new HashMap[Int,(Int,Int,Int)]()
    val coordNode = new HashMap[(Int,Int,Int),Int]()
  for( rIdx <- 0 until xi.rows){
    for( cIdx <- 0 until xi.cols){
      val myIDX = ImageSegmentationDemo.columnMajorIdx(rIdx,cIdx,xi.rows)
      val nextNode = new Node[Vector[Double]](myIDX,xi(rIdx,cIdx).feature,new HashSet[Int]())
      linkCoord.put(myIDX,(rIdx,cIdx,0))
      coordNode.put((rIdx,cIdx,0),nextNode.idx)
      labelVect(myIDX)=yi(rIdx,cIdx).label
      nodeList += nextNode
    }
  }  
  val sortedNodeList = nodeList.sortWith(nodeCmp) //TODO make sure this is increasing order from 
  assert(sortedNodeList.length == xi.rows*xi.cols)
  assert(sortedNodeList(0).idx==0 && sortedNodeList(1).idx==1  && sortedNodeList(2).idx==2)
  val nodeVect = Vector(sortedNodeList.toArray)
   

    linkCoord.keySet.foreach {  key => {
        val coords:(Int,Int,Int) = linkCoord.get(key).get
        nodeVect(key).connections++=coordNode.get((coords._1+1,coords._2,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2+1,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2,coords._3+1))
        nodeVect(key).connections++=coordNode.get((coords._1-1,coords._2,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2-1,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2,coords._3-1))
    } }
  

   
  (new GraphStruct(nodeVect,linkCoord,(xi.rows-1,xi.cols-1,0)), new GraphLabels(Vector(labelVect),numClasses))
}

  def cutSuperPix(dataIn :  D3ArrInt,
      x: Int, y: Int, z: Int, 
      superPixSize: Vector[Int]): Array[Array[Array[Int]]] = {
     
    var out = new Array[Array[Array[Int]]](superPixSize(0))
      for {
        xIdx <- x until (x + superPixSize(0)); yIdx <- y until (y + superPixSize(1)); zIdx <- z until (z + superPixSize(2))
      } {         
        val superX = xIdx - x
        val superY = yIdx - y
        val superZ = zIdx - z
        if( out(superX) == null)
          out(superX) = new Array[Array[Int]](superPixSize(2))
        if ( out(superX)(superY) == null)
          out(superX)(superY) = new Array[Int](superPixSize(3))
        
        out(superX)(superY)(superZ) = dataIn(x)(y)(z)
      }
      out
    }
  
  def simple3dhist(in : Array[Array[Array[Int]]], numBins : Int, histBinSize : Int):Vector[Double]={
      val out = Vector.fill(numBins)(0.0)
      in.view.flatten.flatten.foreach( dat => out(Math.floor((dat-0.000001) / histBinSize).asInstanceOf[Int])+=1)
      out
    }
  
  def graphFrom3dMat(dataIn: Array[Array[Array[Int]]],
                     labels: Array[Array[Array[Int]]],
                     featureFn: Array[Array[Array[Int]]] => Vector[Double],
                     inSuperPixelDim: Vector[Int]): (GraphStruct[Vector[Double], (Int, Int, Int)],GraphLabels) = {

    val xDim = dataIn.length
    val yDim = dataIn(0).length
    val zDim = dataIn(0)(0).length
    assert(xDim == labels.length)
    assert(yDim == labels(0).length)
    assert(zDim == labels(0)(0).length)

    val superPixSize = if (inSuperPixelDim.size == 1)
                          Vector(inSuperPixelDim(0), inSuperPixelDim(0), inSuperPixelDim(0))
                        else {
                          assert(inSuperPixelDim.size == 3)
                          inSuperPixelDim
                        }
                      

    val numSupPixelPerX = floor(xDim / superPixSize(0))
    val numSupPixelPerY = floor(yDim / superPixSize(1))
    val numSupPixelPerZ = floor(zDim / superPixSize(2))

    

    
    assert((xDim - numSupPixelPerX * superPixSize(0)) >= 0)

    def myCutSuperPix(x: Int, y: Int, z: Int): Array[Array[Array[Int]]] = {
      var out = Array.fill(superPixSize(0),superPixSize(1),superPixSize(2)){0}
      for {
        xIdx <- x until (x + superPixSize(0)); yIdx <- y until (y + superPixSize(1)); zIdx <- z until (z + superPixSize(2))
      } {         
        val superX = xIdx - x
        val superY = yIdx - y
        val superZ = zIdx - z
        
        if(x>=dataIn.length||y>=dataIn(0).length||z>=dataIn(0)(0).length)
          return null
        out(superX)(superY)(superZ) = dataIn(x)(y)(z)
        

      }
      out
    }
    
    

    val nodeList = new scala.collection.mutable.ListBuffer[Node[Vector[Double]]]
    var counter =0;
    //val coordLink = new HashMap[(Int,Int,Int),Int]()
    val linkCoord = new HashMap[Int,(Int,Int,Int)]()
    val coordNode = new HashMap[(Int,Int,Int),Int]()
    val labelOut = new LinkedList[Int]()
    for{       supX <- 0 until numSupPixelPerX; supY <- 0 until numSupPixelPerY; supZ <- 0 until numSupPixelPerZ
    } {
      
       val supPixData = myCutSuperPix(supX*superPixSize(0), supY*superPixSize(1), supZ*superPixSize(2))
       val feats = featureFn(supPixData)
       val totalEleNum = superPixSize.reduceLeft( (a1,a2)=>a1*a2).toDouble
       val supPixSum =supPixData.flatMap { x => x }.flatMap{x=>x}.toList.reduceLeft((a1,a2)=>a1+a2)/totalEleNum
       labelOut.append(LinkedList(if(supPixSum>0.5) 1 else 0))//TODO this should be the most freq occuring label. So it generalizes to more than binary 
      //TODO there is a bug in this append statment
       val nextNode = new Node[Vector[Double]](counter,feats,new HashSet[Int]())
       linkCoord.put(counter,(supX,supY,supZ))
       coordNode.put((supX,supY,supZ),nextNode.idx)
       nodeList +=nextNode
       counter+=1
    }
    
    
    val outLabelGraph = new GraphLabels(Vector(labelOut.toArray),2)
    val nodeVect = Vector(nodeList.toArray)
    
    linkCoord.keySet.foreach {  key => {
        val coords = linkCoord.get(key).get
        nodeVect(key).connections++=coordNode.get((coords._1+1,coords._2,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2+1,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2,coords._3+1))
        nodeVect(key).connections++=coordNode.get((coords._1-1,coords._2,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2-1,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2,coords._3-1))
        
      
    } }
    
    

   (new GraphStruct[Vector[Double],(Int,Int,Int)](nodeVect,linkCoord,(xDim-1,yDim-1,zDim-1)), outLabelGraph)
  }
  
  def boundGet[cont](x:Int,y:Int,z:Int,data:Array[Array[Array[cont]]]): Option[cont]={
    if(x>=0 & y >=0 && z >= 0 && x<data.length&&y<data(0).length&&z<data(0)(0).length){
     return Option(data(x)(y)(z))
    }
    else
      return None
  }

  
  //THIS DId not workout how i wanted it 
  def genClusteredGraphData( canvasSize : Int,probUnifRandom: Double, featureNoise : Double, pairRandomItr: Int, numClasses:Int, neighbouringProb : Array[Array[Double]]){
    var unaryData = Array.fill(canvasSize,canvasSize,1)(0)
    for( x <- 0 until unaryData.length){
      for (y <- 0 until unaryData(0).length){
        for(z <- 0 until unaryData(0)(0).length){
           unaryData(x)(y)(z) = (Math.random()*numClasses.toDouble).toInt%numClasses
          
          }
        }
      }
    
   var pairData = Array.fill(canvasSize,canvasSize,1)(0)
     for ( idx <- 0 until pairRandomItr ){
     for( x <- 0 until pairData.length){
      for (y <- 0 until pairData(0).length){
        for(z <- 0 until pairData(0)(0).length){
               
            val neigh = new LinkedList[Vector[Double]]()
            //val self = theData(x)(y)(z)
            var totalProbsPerNeighbouringLabel = Vector(Array.fill(numClasses)(0.0))
            var xN=x; var yN = y ; var zN = z
            if(xN>=0 && yN >=0 && zN >= 0 && xN<pairData.length&&yN<pairData(0).length&&zN<pairData(0)(0).length){
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel+=Vector(neighbouringProb(other))
            }
             xN=x+1;  yN = y ;  zN = z
            if(xN>=0 && yN >=0 && zN >= 0 && xN<pairData.length&&yN<pairData(0).length&&zN<pairData(0)(0).length){
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel+=Vector(neighbouringProb(other))
            }
            xN=x;  yN = y+1 ;  zN = z
            if(xN>=0 && yN >=0 && zN >= 0 && xN<pairData.length&&yN<pairData(0).length&&zN<pairData(0)(0).length){
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel+=Vector(neighbouringProb(other))
            }
            xN=x;  yN = y ;  zN = z+1
            if(xN>=0 && yN >=0 && zN >= 0 && xN<pairData.length&&yN<pairData(0).length&&zN<pairData(0)(0).length){
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel+=Vector(neighbouringProb(other))
            }
            xN=x-1;  yN = y ;  zN = z
            if(xN>=0 && yN >=0 && zN >= 0 && xN<pairData.length&&yN<pairData(0).length&&zN<pairData(0)(0).length){
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel+=Vector(neighbouringProb(other))
            }
            xN=x;  yN = y-1 ;  zN = z
            if(xN>=0 && yN >=0 && zN >= 0 && xN<pairData.length&&yN<pairData(0).length&&zN<pairData(0)(0).length){
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel+=Vector(neighbouringProb(other))
            }
            xN=x;  yN = y ;  zN = z-1
            if(xN>=0 && yN >=0 && zN >= 0 && xN<pairData.length&&yN<pairData(0).length&&zN<pairData(0)(0).length){
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel+=Vector(neighbouringProb(other))
            }
            
            
            
            pairData(x)(y)(z)= randomLabelWithProb(totalProbsPerNeighbouringLabel.toArray)
          }
        }
      }
   
   unaryData = pairData
 //  println("-------------------------------------------------")
   
   }
     println(unaryData.deep.mkString("\n"))
    print("tst")
  }
  
  def randomLabelWithProb (probs : Array[Double] ):Int={
    val totalProbs = probs.toList.reduceLeft((a1,a2)=> a1+a2)
    var randomVal = Math.random()*totalProbs
    for( idx <- 0 until probs.length){ //this can purhaps be done faster with a map or something 
      
      randomVal -= probs(idx) 
      if(randomVal<=0)
        return idx
    }
    return probs.length-1
  }
  
  
  def anotherDataGenFn (canvasSize:Int,portionBackground:Double,numClasses:Int, featureNoise:Double):(GraphStruct[Vector[Double], (Int, Int, Int)],GraphLabels) ={
    //Gen random mat of 10x10 
      //Force it to be mostly zeros by rounding down after 0.8
      //Per 1 that still exists choose one of the non zero classes to replace it with
    //Generate a much larger 100x100 where each pixel of original mat represents 10x10 pixes of that value
    //Add Random noise 
    //Generate a graph with excisting functions and make the superpixel size 5x5 
    assert(canvasSize%4==0)
    val random = new java.security.SecureRandom           
    val topLvl = Array.fill(canvasSize/4,canvasSize/4){ if(random.nextDouble()<portionBackground) 0 else 1}
    val scaled = DenseMatrix.zeros[Int](canvasSize,canvasSize)
    
    //Lets go directly to a graph
    for( x <-0 until canvasSize/4){
      for(y <-0 until canvasSize/4){
        if(topLvl(x)(y)!=0){
          val curClass = random.nextInt(numClasses-1)+1
          scaled(x*4 to x*4+3,y*4 to y*4+3):=curClass
        }
      }
    }
    
    def featureFn(label:Int):Vector[Double]={
      val outF:Array[Double] =if( random.nextDouble()>featureNoise){
      val tmp = Array.fill[Double](numClasses){random.nextDouble()}
      tmp(label)=1.0
      tmp
      }
      else{
        Array.fill(numClasses){random.nextDouble()}
      }
      Vector(outF)
    }
    
    val nodeList = new scala.collection.mutable.ListBuffer[Node[Vector[Double]]]
    var counter =0;
    //val coordLink = new HashMap[(Int,Int,Int),Int]()
    val linkCoord = new HashMap[Int,(Int,Int,Int)]()
    val coordNode = new HashMap[(Int,Int,Int),Int]()
    val labelOut =  new ListBuffer[Int]
    for{       supX <- 0 until canvasSize; supY <- 0 until canvasSize
    } {
      
       val feats = featureFn(scaled(supX,supY))

       labelOut++=LinkedList(scaled(supX,supY))
       val nextNode = new Node[Vector[Double]](counter,feats,new HashSet[Int]())
       linkCoord.put(counter,(supX,supY,0))
       coordNode.put((supX,supY,0),nextNode.idx)
       nodeList += nextNode
       counter  += 1
    }
    
        
    val outLabelGraph = new GraphLabels(Vector(labelOut.toArray),numClasses)
    val nodeVect = Vector(nodeList.toArray)
    
    
    linkCoord.keySet.foreach {  key => {
        val coords = linkCoord.get(key).get
        nodeVect(key).connections++=coordNode.get((coords._1+1,coords._2,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2+1,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2,coords._3+1))
        nodeVect(key).connections++=coordNode.get((coords._1-1,coords._2,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2-1,coords._3))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2,coords._3-1))
        
      
    } }
    
    

   (new GraphStruct[Vector[Double],(Int,Int,Int)](nodeVect,linkCoord,(canvasSize-1,canvasSize-1,0)), outLabelGraph)
 
  
  }
  
  
  def genSquareBlobs( howMany:Int,anvasSize:Int,portionBackground:Double,numClasses:Int, featureNoise:Double ):Seq[LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels]]={
    val out = for( i <- 0 until howMany) yield{
      val (xGraph,yList) = anotherDataGenFn(anvasSize,portionBackground,numClasses, featureNoise)
         new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](yList,xGraph)
    }
    out
  }
  

  
  
  
}