package ch.ethz.dalab.dissolve.examples.neighbourhood

import ch.ethz.dalab.dissolve.examples.imgseg3d._
import ch.ethz.dalab.dissolve.examples.neighbourhood._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.max
import breeze.linalg.min
import breeze.linalg.normalize
import breeze.linalg.Vector
import breeze.math._
import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object testGraph {

  def main(args: Array[String]): Unit = {
       val someD3 = GraphUtils.d3randomVecInt()
       val aHist =  GraphUtils.simple3dhist(someD3,5,(Math.floor(255/5).asInstanceOf[Int]))
      //  val moreD3 = Array.fill(100,100,100){(Math.random()*255).asInstanceOf[Int]}
       val pairList = ThreeDimUtils.generateSomeBalls(1, 10, 2, 0.0)
       def someFeat (in : Array[Array[Array[Int]]]):Vector[Double]={
            GraphUtils.simple3dhist(in,5,255/5)
          }
        val aGraph = GraphUtils.graphFrom3dMat(pairList(0)._1,pairList(0)._2,someFeat,Vector(5,5,5))
        print("fun")  
        
        //TODO remove this debug stuff 
        val some3dimg = Array.fill(10,20,1){Math.round(5*Math.random()).asInstanceOf[Int]}
        val some3dLabels = Array.fill(10,20,1){Math.round(100*Math.random()).asInstanceOf[Int]}
        val bGraph= GraphUtils.graphFrom3dMat(some3dimg,some3dLabels,someFeat,Vector(5,5,5))
        //
        
        
        //Read in some real data 
        val conf = new SparkConf().setAppName("GraphTest").setMaster("local")
        val sc = new SparkContext(conf)
        val rawCsv = sc.textFile("../data/calcifications/combined_image_with_labels.csv")
        val header = rawCsv.first
        // transform the data from a string to a case class
        case class mcdPoint(x: Int, y: Int, abs: Double, dci: Double, cal: Double)
        
        def mcdFromString(inStr: String) = {
          val strArr = inStr.split(",")
          mcdPoint(strArr(0).toInt,strArr(1).toInt,
          strArr(2).toDouble,strArr(3).toDouble,
          strArr(4).toDouble)
        }
        // remove header, split columns
        val mcdData = rawCsv.filter(_(0)!='x').map(mcdFromString(_)).cache
        val statFun = mcdData.map(_.abs).stats()
        print(statFun)
        // res8: org.apache.spark.util.StatCounter = (count: 1440000, mean: 0.460314, stdev: 0.060376, max: 0.678339, min: 0.345074)
        val histFun = mcdData.map(_.abs).histogram(20)
        print(histFun)
        print("fun")  
        
        }
}