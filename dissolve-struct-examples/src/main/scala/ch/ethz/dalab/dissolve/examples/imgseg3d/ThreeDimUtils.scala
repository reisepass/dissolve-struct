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
import scala.collection.mutable.HashMap
import scala.util.Random
import scala.collection.mutable.HashSet
import breeze.stats.DescriptiveStats._
import scala.collection.mutable.ListBuffer

//TODO the metaData is hacky, think of something nicer.
class ThreeDimMat[DataType](dimPerCoord: Vector[Int], metaData: HashMap[String, Object] = new HashMap[String, Object](),
                            initVecDat: Array[DataType] = null, initMatData: Array[Array[Array[DataType]]] = null, isNominal: Boolean = false, classes: Set[DataType] = null)(implicit m: ClassManifest[DataType]) extends Serializable {

  def meta = metaData
  val dims = if (dimPerCoord.length == 1) new DenseVector[Int](Array(dimPerCoord(0), dimPerCoord(0), dimPerCoord(0))) else dimPerCoord
  def xDim = dims(0)
  def yDim = dims(1)
  def zDim = dims(2)
  var myData: Array[Array[Array[DataType]]] = new Array[Array[Array[DataType]]](dims(0))
  assert(dims.size == 3)
  for (x <- 0 until dims(0)) {
    myData(x) = new Array[Array[DataType]](dims(1))
    for (y <- 0 until dims(1)) {
      myData(x)(y) = new Array[DataType](dims(2))
    }
  }
  if (initVecDat != null) {
    assert(initMatData == null)
    assert(initVecDat.length == dims(0) * dims(1) * dims(2))
    reshape(initVecDat)
  }
  if (initMatData != null) {
    assert(initVecDat == null)
    assert(initMatData.length == dims(0))
    assert(initMatData(0).length == dims(1))
    assert(initMatData(0)(0).length == dims(2))
    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          myData(x)(y)(z) = initMatData(x)(y)(z);
        }
      }
    }

  }

  var classFreq = new HashMap[DataType, Double]
  if (isNominal) {
    assert(classes != null)
    classFreq = findAllClasses()
  }
  def frequencies = if (isNominal) classFreq else null
  def classSet = if (isNominal) classes else null
  def nominal = this.isNominal
  if (isNominal) assert(classSet.equals((0 until classSet.size toList).toSet))

  def reshape(inData: Array[DataType]) {
    reshape(new DenseVector(inData))
  }
  def reshape(inData: Vector[DataType]) {
    var vCounter = 0;
    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          myData(x)(y)(z) = inData(vCounter)
          vCounter += 1
        }
      }
    }
  }
  def melt(): Array[DataType] = {

    var outVector = new Array[DataType](dims(0) * dims(1) * dims(2))
    var vCounter = 0
    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          outVector(vCounter) = myData(x)(y)(z)
          vCounter += 1
        }
      }
    }
    outVector
  }

  def get(x: Int, y: Int, z: Int): DataType = {
    myData(x)(y)(z)
  }
  def set(x: Int, y: Int, z: Int, newVal: DataType) {
    myData(x)(y)(z) = newVal
  }

  def equals(other: ThreeDimMat[DataType]): Boolean = {

    if (this.xDim != other.xDim)
      return false
    if (this.yDim != other.yDim)
      return false
    if (this.zDim != other.zDim)
      return false

    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          if (this.get(x, y, z) != other.get(x, y, z))
            return false
        }
      }
    }

    return true
  }

  def dimAgree[G >: DataType](other: ThreeDimMat[G]): Boolean = {
    if (other.xDim != this.xDim)
      return false
    if (other.yDim != this.yDim)
      return false
    if (other.zDim != this.zDim)
      return false
    true
  }

  //Warning, if this method is evoced on non Nominal data it will douplicate the entire dataset 

  def findAllClasses(): HashMap[DataType, Double] = {
    assert(isNominal)
    // var classSet = new HashSet[DataType];  
    var classCount = new HashMap[DataType, Int]
    classes.map(C => classCount.put(C, 0))
    var totalCount = 0;
    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          classCount.update(this.get(x, y, z), classCount.getOrElse(this.get(x, y, z), 0) + 1)
          totalCount += 1
        }
      }
    }

    var classFreq = new HashMap[DataType, Double]
    classCount.foreach { case (key, value) => classFreq.put(key, (value.asInstanceOf[Double] / totalCount)) }
    classFreq

  }

}

class NominalThreeDimMat[DataType](dimPerCoord: Vector[Int], initVecDat: Array[DataType] = null,
                                   initMatData: Array[Array[Array[DataType]]] = null, 
                                   classes: Set[DataType])
                                   (implicit m: ClassManifest[DataType])
                                    extends Serializable {

  val dims = if (dimPerCoord.length == 1) new DenseVector[Int](Array(dimPerCoord(0), dimPerCoord(0), dimPerCoord(0))) else dimPerCoord
  def xDim = dims(0)
  def yDim = dims(1)
  def zDim = dims(2)
  var myData: Array[Array[Array[DataType]]] = new Array[Array[Array[DataType]]](dims(0))
  assert(dims.size == 3)
  for (x <- 0 until dims(0)) {
    myData(x) = new Array[Array[DataType]](dims(1))
    for (y <- 0 until dims(1)) {
      myData(x)(y) = new Array[DataType](dims(2))
    }
  }
  if (initVecDat != null) {
    assert(initMatData == null)
    assert(initVecDat.length == dims(0) * dims(1) * dims(2))
    reshape(initVecDat)
  }
  if (initMatData != null) {
    assert(initVecDat == null)
    assert(initMatData.length == dims(0))
    assert(initMatData(0).length == dims(1))
    assert(initMatData(0)(0).length == dims(2))
    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          myData(x)(y)(z) = initMatData(x)(y)(z);
        }
      }
    }

  }

  var classFreq = new HashMap[DataType, Double]

  assert(classes != null)
  classFreq = findAllClasses()

  def frequencies = classFreq
  def classSet = classes
  def nominal = true
  assert(classSet.equals((0 until classSet.size toList).toSet))

  def reshape(inData: Array[DataType]) {
    reshape(new DenseVector(inData))
  }
  def reshape(inData: Vector[DataType]) {
    var vCounter = 0;
    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          myData(x)(y)(z) = inData(vCounter)
          vCounter += 1
        }
      }
    }
  }
  def melt(): Array[DataType] = {

    var outVector = new Array[DataType](dims(0) * dims(1) * dims(2))
    var vCounter = 0
    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          outVector(vCounter) = myData(x)(y)(z)
          vCounter += 1
        }
      }
    }
    outVector
  }

  def get(x: Int, y: Int, z: Int): DataType = {
    myData(x)(y)(z)
  }
  def set(x: Int, y: Int, z: Int, newVal: DataType) {
    myData(x)(y)(z) = newVal
  }

  def equals(other: ThreeDimMat[DataType]): Boolean = {

    if (this.xDim != other.xDim)
      return false
    if (this.yDim != other.yDim)
      return false
    if (this.zDim != other.zDim)
      return false

    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          if (this.get(x, y, z) != other.get(x, y, z))
            return false
        }
      }
    }

    return true
  }

  def dimAgree[G >: DataType](other: ThreeDimMat[G]): Boolean = {
    if (other.xDim != this.xDim)
      return false
    if (other.yDim != this.yDim)
      return false
    if (other.zDim != this.zDim)
      return false
    true
  }

  //Warning, if this method is evoced on non Nominal data it will douplicate the entire dataset 

  def findAllClasses(): HashMap[DataType, Double] = {

    // var classSet = new HashSet[DataType];  
    var classCount = new HashMap[DataType, Int]
    classes.map(C => classCount.put(C, 0))
    var totalCount = 0;
    for (x <- 0 until dims(0)) {
      for (y <- 0 until dims(1)) {
        for (z <- 0 until dims(2)) {
          classCount.update(this.get(x, y, z), classCount.getOrElse(this.get(x, y, z), 0) + 1)
          totalCount += 1
        }
      }
    }

    var classFreq = new HashMap[DataType, Double]
    classCount.foreach { case (key, value) => classFreq.put(key, (value.asInstanceOf[Double] / totalCount)) }
    classFreq

  }

}

/*
//TODO I dont know how to make a variable size Matrix out of arrays. the issue is I need to specify the Array[]type without knowing it 
class HighDimMat[MatStruct, DataType](numCoords: Int, dimPerCoord: Vector[Int], metaData: HashMap[String, Object]) {
  var myData: MatStruct

}
* 
*/

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
      } myMatrix(x)(y)(z) = min(max((myMatrix(x)(y)(z) + Math.abs(howMuchNoise * (Math.random() * 255) - Math.abs(howMuchNoise * (Math.random() * 255))).asInstanceOf[Int]), 1), 255);
    }

    (myMatrix, myLabels)
  }

  def unsignedByte(in: Int): Int = {
    in & 0xFF
  }

  //TODO this function is not yet normalzied 
  //                                                                                       
  def hist3d(dataIn: Array[Array[Array[Int]]], numBins: Int, superPixSize: Int): ThreeDimMat[Array[Double]] = {
    val xDim = dataIn.length;
    val yDim = dataIn(0).length;
    val zDim = dataIn(0)(0).length;

    assert(xDim == yDim && yDim == zDim) //TODO this may not be needed 

    //var outHist = new Array[Array[Array[Array[Int]]]](xDim)

    val numSupPixelPerX = floor(xDim / superPixSize)

    val numSupPixelPerY = floor(yDim / superPixSize)
    val numSupPixelPerZ = floor(zDim / superPixSize)

    var outHist = new ThreeDimMat[Array[Double]](Vector(numSupPixelPerX, numSupPixelPerY, numSupPixelPerZ))

    val extraPix = xDim - numSupPixelPerX * superPixSize
    assert(extraPix >= 0)

    val histBinSize = 255 / numBins //TODO verify with dataformat, throw exception of not max value 

    def patchHist(x: Int, y: Int, z: Int): Array[Double] = {
      var localHist: Array[Double] = Array.fill(numBins)(0.0);
      for {
        xIdx <- x until (x + superPixSize); yIdx <- y until (y + superPixSize); zIdx <- 0 until (z + superPixSize)
      } {
        var insertIDX = floor((dataIn(xIdx)(yIdx)(zIdx)) / histBinSize).asInstanceOf[Int]
        if (insertIDX == numBins)
          insertIDX = insertIDX - 1;
        if (floor((dataIn(xIdx)(yIdx)(zIdx)) / histBinSize).asInstanceOf[Int] > localHist.length) {
          val primDebug = dataIn(xIdx)(yIdx)(zIdx)
          val prim2 = (dataIn(xIdx)(yIdx)(zIdx))
          val prim3 = (Int.MaxValue)
          print(numBins)
        }
        if (floor((dataIn(xIdx)(yIdx)(zIdx)) / histBinSize).asInstanceOf[Int] < 0) {
          print(numBins)
        }
        println(floor((dataIn(xIdx)(yIdx)(zIdx)) / histBinSize).asInstanceOf[Int])
        localHist(insertIDX) += 1.0
      }
      localHist

    }

    for {
      x <- 0 until numSupPixelPerX; y <- 0 until numSupPixelPerY; z <- 0 until numSupPixelPerZ //TODO change this to lazy sequence 
    } outHist.set(x, y, z, patchHist(x, y, z))

    outHist
  }

  def superLabels3d(dataIn: Array[Array[Array[Int]]], superPixSize: Int, labelMapping: (Array[Int]) => Int, numYclasses: Int = 2): NominalThreeDimMat[Int] = {
    val xDim = dataIn.length
    val yDim = dataIn(0).length
    val zDim = dataIn(0)(0).length

    //TODO Bookmark
    //I need to include the class labels into the y Training data. I think i can do this by searching the output space of labelMapping since input is just 0-255 

    val numSupPixelPerX = floor(xDim / superPixSize)

    val numSupPixelPerY = floor(yDim / superPixSize)
    val numSupPixelPerZ = floor(zDim / superPixSize)

    var out = new NominalThreeDimMat[Int](Vector(numSupPixelPerX, numSupPixelPerY, numSupPixelPerZ), classes = (0 until numYclasses toList).toSet)

    def patchy(x: Int, y: Int, z: Int): Array[Int] = {
      var out = new ListBuffer[Int]()
      for {
        xIdx <- superPixSize*x until (superPixSize*x + superPixSize); 
        yIdx <- superPixSize*y until (superPixSize*y + superPixSize); 
        zIdx <- superPixSize*z until (superPixSize*z + superPixSize)
      } {
        out.append(dataIn(xIdx)(yIdx)(zIdx))
      }
      if(out.length<2)
        print("wtf")
      out.toArray
    }

    for {
      x <- 0 until numSupPixelPerX; y <- 0 until numSupPixelPerY; z <- 0 until numSupPixelPerZ //TODO change this to lazy sequence 
    } out.set(x, y, z, labelMapping(patchy(x, y, z)))

    out
  }

  def generateSomeBalls(howMany: Int, canvisSize: Int, ballRadius: Double = -1, howMuchNoise: Double = 0.1): Array[(Array[Array[Array[Int]]], Array[Array[Array[Int]]])] = {

    var out: Array[(Array[Array[Array[Int]]], Array[Array[Array[Int]]])] = new Array[(Array[Array[Array[Int]]], Array[Array[Array[Int]]])](howMany)
    val radius = if (ballRadius < 0) canvisSize / 3 else ballRadius

    for (i <- 0 until howMany) {

      val centerX = round(Math.random() * canvisSize)
      val centerY = round(Math.random() * canvisSize)
      val centerZ = round(Math.random() * canvisSize)
      val (ball, ballLabel) = genSomeShapesGrey(canvisSize, howMuchNoise, (x: Int, y: Int, z: Int) => if (Math.sqrt(((centerX - x) ^ 2 + (centerY - y) ^ 2 + (centerZ - z) ^ 2)) > radius) 1 else 255)

      out(i) = (ball, ballLabel)
    }

    out
  }

  def mean(dataIn: Array[Int]): Double = {

    var cumsum = 0
    for (x <- 0 until dataIn.length) {
      cumsum += dataIn(x)
    }
    return cumsum / dataIn.length
  }

  def generateSomeData(howMany: Int, canvisSize: Int, numBins: Int, superPixSize: Int, howMuchNoise: Double = 0.1): (Array[LabeledObject[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]]], Array[LabeledObject[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]]]) = {
    val trainDataRaw = generateSomeBalls(howMany, canvisSize, howMuchNoise);
    val testDataRaw =  generateSomeBalls(howMany, canvisSize, howMuchNoise);
    val trainingData = trainDataRaw.map(T => LabeledObject(
                superLabels3d(T._2, superPixSize, TMP => if (this.mean(TMP) > 0.5) 1 else 0), 
                hist3d(T._1, numBins, superPixSize)))
    val testData = testDataRaw.map(T => LabeledObject(
                superLabels3d(T._2, superPixSize, TMP => if (this.mean(TMP) > 0.5) 1 else 0), 
                hist3d(T._1, numBins, superPixSize)))
    return (trainingData, testData)
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

    val funRandom = Seq.fill(10 * 10 * 10)(Random.nextInt(10)).toArray
    var thrM = new ThreeDimMat[Int](Vector(10, 10, 10), initVecDat = funRandom)
    val meltM = thrM.melt()
    var thrRec = new ThreeDimMat[Int](Vector(10, 10, 10), initVecDat = meltM, isNominal = true)
    assert(thrM.equals(thrRec))

    var outSum: Double = 0;
    thrRec.classFreq.foreach { case (key, value) => outSum += value }
    thrRec.classFreq.foreach { case (key, value) => println(" k" + key + " -> " + value) }

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
    val xSet = Array[Int](0, 1, 2)

    val histTest = hist3d(cubeMid, 3, 5)
    val histTest2 = hist3d(hollowBall, 3, 5)
    val histTest3 = hist3d(ball, 3, 5)
    println(cooredSum)

  }
}