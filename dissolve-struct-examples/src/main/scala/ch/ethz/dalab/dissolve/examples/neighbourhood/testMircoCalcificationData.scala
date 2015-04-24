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
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import ch.ethz.dalab.dissolve.examples.neighbourhood.CSVParboiledParserCLI
import org.parboiled2.ParserInput

object testMircoCalcificationData {

  def main(args: Array[String]): Unit = {
    val someD3 = GraphUtils.d3randomVecInt()
    val aHist = GraphUtils.simple3dhist(someD3, 5, (Math.floor(255 / 5).asInstanceOf[Int]))
    //  val moreD3 = Array.fill(100,100,100){(Math.random()*255).asInstanceOf[Int]}

    print("fun")

    lazy val inputfile: ParserInput = scala.io.Source.fromFile("../data/calcifications/combined_image_with_labels.csv").mkString
    val parser = new CSVParboiledParserSimple(inputfile)
    val fun = parser.doStuff
    if (fun.isSuccess) {
      val parseTree = fun.get
      var t0 = System.nanoTime()
      val lastRow = parseTree.last
      val maxX = lastRow(0).toInt
      val maxY = lastRow(1).toInt
      var t1 = System.nanoTime()
      println("Elapsed time: " + (t1 - t0)/Math.pow(10,9) + "s")
      
      
      t0 = System.nanoTime()
      val absData = DenseMatrix.zeros[Double](maxX, maxY)
      val dciData = DenseMatrix.zeros[Double](maxX, maxY)
      val calData = DenseMatrix.zeros[Int](maxX, maxY)
      t1 = System.nanoTime()
      println("2Elapsed time: " + (t1 - t0)/Math.pow(10,9) + "s")
      t0 = System.nanoTime()
      parseTree.tail.foreach { curRow =>
        {
          val xI = curRow(0).toInt
          val yI = curRow(1).toInt
          absData(xI-1, yI-1) = curRow(2).toDouble
          dciData(xI-1, yI-1) = curRow(3).toDouble
          dciData(xI-1, yI-1) = curRow(4).toDouble
        }
       
      }
 t1 = System.nanoTime()
        println("3Elapsed time: " + (t1 - t0)/Math.pow(10,9) + "s")
    }
    //Read in some real data 
    val conf = new SparkConf().setAppName("GraphTest").setMaster("local")
    val sc = new SparkContext(conf)
    val rawCsv = sc.textFile("../data/calcifications/combined_image_with_labels.csv")
    val header = rawCsv.first
    // transform the data from a string to a case class
    case class mcdPoint(x: Int, y: Int, abs: Double, dci: Double, cal: Double)

    def mcdFromString(inStr: String) = {
      val strArr = inStr.split(",")
      mcdPoint(strArr(0).toInt, strArr(1).toInt,
        strArr(2).toDouble, strArr(3).toDouble,
        strArr(4).toDouble)
    }
    // remove header, split columns
    val mcdData = rawCsv.filter(_(0) != 'x').map(mcdFromString(_)).cache
    val statFun = mcdData.map(_.abs).stats()
    print(statFun)
    // res8: org.apache.spark.util.StatCounter = (count: 1440000, mean: 0.460314, stdev: 0.060376, max: 0.678339, min: 0.345074)
    val histFun = mcdData.map(_.abs).histogram(20)
    print(histFun)
    print("fun")

  }
}