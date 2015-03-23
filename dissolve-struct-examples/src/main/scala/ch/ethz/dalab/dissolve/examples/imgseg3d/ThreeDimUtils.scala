package ch.ethz.dalab.dissolve.examples.imgseg3d
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




object ThreeDimUtils {
  val LABEL_PIVOTE = 50;
  def genSomeShapesGrey(size: Int, howMuchNoise: Double, shapeFunc: (Int, Int, Int) => Int): (Array[Array[Array[Int]]], Array[Array[Array[Int]]]) = {

    var myMatrix = Array.ofDim[Int](size, size, size);

    //TODO change this to some nice map (sapply) call
    val idxs = 0 until myMatrix.length
    for {
      x <- idxs; y <- idxs; z <- idxs
    } myMatrix(x)(y)(z) = shapeFunc(x, y, z)

    var myLabels = Array.ofDim[Int](size, size, size);
    for {
      x <- idxs; y <- idxs; z <- idxs //TODO change this to lazy sequence 
    } myLabels(x)(y)(z) = if (myMatrix(x)(y)(z) > LABEL_PIVOTE) 1 else 0

    if (howMuchNoise > 0) {
      for {
        x <- idxs; y <- idxs; z <- idxs //TODO change this to lazy sequence 
      } myMatrix(x)(y)(z) = min(max( (myMatrix(x)(y)(z) + Math.abs(howMuchNoise * (Math.random() * 255) - Math.abs(howMuchNoise * (Math.random() * 255)) ).asInstanceOf[Int]) , 1),255);
    }

    (myMatrix, myLabels)
  }
  
  
  def unsignedByte( in : Int) : Int = { 
    in & 0xFF
  }
  
//                                                                                       x     y     z      hist
  def hist3d (dataIn : Array[Array[Array[Int]]], numBins : Int, superPixSize : Int) : Array[Array[Array[Array[Int]]]] = {
    val xDim = dataIn.length; 
    val yDim = dataIn(0).length;
    val zDim = dataIn(0)(0).length; 
    
    assert(xDim == yDim && yDim == zDim) //TODO this may not be needed 
    
    //var outHist = new Array[Array[Array[Array[Int]]]](xDim)
    
    
    val numSupPixelPerX =  floor(xDim / superPixSize)
  
    val numSupPixelPerY = floor(yDim / superPixSize)
    val numSupPixelPerZ = floor(zDim / superPixSize)
    
    var outHist = Array.fill(numSupPixelPerX, numSupPixelPerY ,numSupPixelPerZ, numBins)(0)
    
    val extraPix = xDim - numSupPixelPerX*superPixSize
    assert( extraPix>=0)
    
    val histBinSize = 255/numBins  //TODO verify with dataformat, throw exception of not max value 
              
    
    def patchHist ( x: Int, y : Int,z :Int ): Array[Int] = {
      var localHist = new Array[Int](numBins);
      for {
        xIdx <- x until (x+numSupPixelPerX); yIdx <- y until (y+numSupPixelPerX); zIdx <-0 until (z+numSupPixelPerX) 
      } { 
        var insertIDX = floor((dataIn(xIdx)(yIdx)(zIdx))/histBinSize).asInstanceOf[Int] 
        if(insertIDX == numBins)
          insertIDX = insertIDX -1; 
          if( floor((dataIn(xIdx)(yIdx)(zIdx))/histBinSize).asInstanceOf[Int] > localHist.length){
             val primDebug =dataIn(xIdx)(yIdx)(zIdx)
             val prim2 = (dataIn(xIdx)(yIdx)(zIdx))
             val prim3 =(Int.MaxValue)
            print ( numBins)
          }
          if(floor( (dataIn(xIdx)(yIdx)(zIdx))/histBinSize).asInstanceOf[Int] <0){
            print(numBins)
          }
           println(floor( (dataIn(xIdx)(yIdx)(zIdx))/histBinSize).asInstanceOf[Int])
          localHist( insertIDX) += 1
        }
      localHist
      
    }
    
    
    for {
        x <- 0 until numSupPixelPerX; y <- 0 until numSupPixelPerY; z <- 0 until numSupPixelPerZ //TODO change this to lazy sequence 
      } outHist(x)(y)(z)= patchHist(x,y,z)

      
      outHist
  }
  
  
  
  /*
  //TODO Bookmark Next creat the histogram features on these grey scale arrays 
  def genSome3dData(size: Int): (Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]], Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]]) = {
    val canvisSize = size;
    val radius = canvisSize / 4
    val centerX, centerY, centerZ = floor(canvisSize / 2)
    val innerRadius = canvisSize / 4 - canvisSize / 5
    val (hollowBall, hollowBallLabel) = genSomeShapesGrey(size, 0, (x: Int, y: Int, z: Int) => if (Math.sqrt(((centerX - x) ^ 2 + (centerY - y) ^ 2 + (centerZ - z) ^ 2)) < radius && Math.sqrt(((centerX - x) ^ 2 + (centerY - y) ^ 2 + (centerZ - z) ^ 2)) > innerRadius) Int.MaxValue else 1)

  }
  * 
  */

  def main(args: Array[String]): Unit = {

    val canvisSize = 15;
    val (cooredSum, cooredSumLabel) = genSomeShapesGrey(canvisSize, 0, (x: Int, y: Int, z: Int) => (x + y + z).asInstanceOf[Int])
    val radius = canvisSize / 4
    val centerX, centerY, centerZ = floor(canvisSize / 2)

    val (ball, ballLabel) = genSomeShapesGrey(canvisSize, 0.5, (x: Int, y: Int, z: Int) => if (Math.sqrt(((centerX - x) ^ 2 + (centerY - y) ^ 2 + (centerZ - z) ^ 2)) > radius) 1 else 255) //TODO lookup vector notation for this euclidian distance
    val innerRadius = canvisSize / 4 - canvisSize / 5
    val (hollowBall, hollowBallLabel) = genSomeShapesGrey(canvisSize, 0, (x: Int, y: Int, z: Int) => if (Math.sqrt(((centerX - x) ^ 2 + (centerY - y) ^ 2 + (centerZ - z) ^ 2)) < radius && Math.sqrt(((centerX - x) ^ 2 + (centerY - y) ^ 2 + (centerZ - z) ^ 2)) > innerRadius) 255 else 1)

    val sideLeng = canvisSize / 3;
    val cubeFullTopLeft = genSomeShapesGrey(canvisSize, 0, (x, y, z) => if (x < sideLeng && y < sideLeng && z < sideLeng) 1 else 0)
    val (cubeMid, cubeMidLabel) = genSomeShapesGrey(canvisSize, 1, (x, y, z) => if (Math.abs(centerX - x) < sideLeng / 2 && Math.abs(centerY - y) < sideLeng / 2 && Math.abs(centerZ - z) < sideLeng / 2) 255 else 1)
 val xSet =   Array[Int](0,1,2)
  
    
    
    val histTest = hist3d(cubeMid, 3, 5) 
    val histTest2 = hist3d(hollowBall, 3, 5) 
    val histTest3 = hist3d(ball, 3, 5) 
    println(cooredSum)

  }
}