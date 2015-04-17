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


class GraphStruct[Features, OriginalCoord](graph: Vector[Node[Features]],
                                           dataLink: HashMap[Int, OriginalCoord]) {

  def graphNodes = graph
  def dataGraphLink = dataLink

}

case class Node[Features](
  val idx: Int,
  val features: Features,
  val connections: scala.collection.mutable.Set[Node[Features]]) extends Serializable {

}



object GraphUtils {
  type D3ArrInt = Array[Array[Array[Int]]]
  type D3ArrDbl = Array[Array[Array[Double]]]
      def randomVec () : Array[Double]={
      Array.fill[Double](10){Math.random()}
     
    }
def d3randomVecDlb () : D3ArrDbl={
      Array.fill[Double](10,10,10){Math.random()}
     
    }
def d3randomVecInt (x:Int =10, y:Int = 10, z:Int= 10) : D3ArrInt={
      Array.fill(x,y,z){(Math.random()*255).asInstanceOf[Int]}
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
      in.view.flatten.flatten.foreach( dat => out((dat / histBinSize).asInstanceOf[Int])+=1)
      out
    }
  
  def graphFrom3dMat(dataIn: Array[Array[Array[Int]]],
                     labels: Array[Array[Array[Int]]],
                     featureFn: Array[Array[Array[Int]]] => Vector[Double],
                     inSuperPixelDim: Vector[Int]): GraphStruct[Vector[Double], (Int, Int, Int)] = {

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
        
        if(x>=dataIn.length||y>=dataIn(0).length||z>=dataIn(0)(0).length)
          return null
        out(superX)(superY)(superZ) = dataIn(x)(y)(z)
        

      }
      out
    }
    
    
    
    

    
    def someFeat (in : Array[Array[Array[Int]]]):Vector[Double]={
      simple3dhist(in,5,255/5)
    }
    
    simple3dhist(Array.fill[Int](10,10,10){0}, 5, 255 / 5)
    
    var nodeList = new LinkedList[Node[Vector[Double]]]
    var counter =0;
    //val coordLink = new HashMap[(Int,Int,Int),Int]()
    val linkCoord = new HashMap[Int,(Int,Int,Int)]()
    val coordNode = new HashMap[(Int,Int,Int),Node[Vector[Double]]]()
    for{       supX <- 0 until numSupPixelPerX; supY <- 0 until numSupPixelPerY; supZ <- 0 until numSupPixelPerZ
    } {
      
       val supPixData = myCutSuperPix(supX*superPixSize(0), supY*superPixSize(1), supZ*superPixSize(2))
       val feats = someFeat(supPixData)
       counter+=1

       val nextNode = new Node[Vector[Double]](counter,feats,new HashSet[Node[Vector[Double]]]())
       linkCoord.put(counter,(supX,supY,supZ))
       coordNode.put((supX,supY,supZ),nextNode)
       nodeList append LinkedList(nextNode)
    }
    val nodeVect = Vector(nodeList.toArray)
    
    linkCoord.keySet.foreach {  key => {
        val coords = linkCoord.get(key).get
        nodeVect(key).connections++=coordNode.get((coords._1+1,coords._2,coords._2))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2+1,coords._2))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2,coords._2+1))
        nodeVect(key).connections++=coordNode.get((coords._1-1,coords._2,coords._2))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2-1,coords._2))
        nodeVect(key).connections++=coordNode.get((coords._1,coords._2,coords._2-1))
        
      
    } }
    

   new GraphStruct[Vector[Double],(Int,Int,Int)](nodeVect,linkCoord)
  }
}