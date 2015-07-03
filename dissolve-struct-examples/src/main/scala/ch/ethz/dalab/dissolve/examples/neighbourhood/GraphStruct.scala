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
import java.util.concurrent.atomic.AtomicInteger
import java.awt.Color
import ij.io.Opener
import java.awt.FontMetrics
import java.awt.Font
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;

class GraphStruct[Features, OriginalCoord](graph: Vector[Node[Features]],
                                           originalDataMappingFilePath: String) extends Serializable { //dataLink could be a Vector, no reason for hashMap

  def graphNodes = graph
  def originMapFile = originalDataMappingFilePath
  //TODO Add?? mapping from original xyz to nodes. For prediction we need something like this but it could be more efficient to just have a print out method that returns a xyz matrix 
  def getF(i: Int): Features = { graphNodes(i).features }
  def get(i: Int): Node[Features] = { graphNodes(i) }
  def getC(i: Int): scala.collection.mutable.Set[Int] = { graphNodes(i).connections }
  def size = graphNodes.size

}

case class Node[Features](
  val idx: Int,
  val features: Features,
  val connections: scala.collection.mutable.Set[Int]) extends Serializable {
  override def hashCode(): Int = {
    idx
  }

}

case class GraphLabels(d: Vector[Int], numClasses: Int, originalLabelFile: String = "None") extends Serializable {
  assert(numClasses > 0)
  def isInverseOf(other: GraphLabels): Boolean = {
    if (other.d.size != d.size)
      return false
    else {
      for (i <- 0 until d.size) {
        if (other.d(i) == d(i))
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
  def randomVec(): Array[Double] = {
    Array.fill[Double](10) { Math.random() }

  }
  val curTime = System.currentTimeMillis()

  val someColorsRGB = List((240, 163, 255), (0, 117, 220), (153, 63, 0), (76, 0, 92), (25, 25, 25), (0, 92, 49), (43, 206, 72), (255, 204, 153), (128, 128, 128), (148, 255, 181), (143, 124, 0), (157, 204, 0), (194, 0, 136), (0, 51, 128), (255, 164, 5), (255, 168, 187), (66, 102, 0), (255, 0, 16), (94, 241, 242), (0, 153, 143), (224, 255, 102), (116, 10, 255), (153, 0, 0), (255, 255, 128), (255, 255, 0), (255, 80, 5))
  val someColors = List(0x9C2141, 0x6AF23A, 0x4BCCDB, 0x8B66EE, 0xF7BE24, 0x235B1D, 0x253C6F, 0xE9B79D, 0xF2531C, 0xE033A5, 0xACE89F, 0x86530D, 0xE2ACDE, 0x3D2F24, 0x4EA1EC, 0x8BB630, 0xEE3FF2, 0x788388, 0x701E68, 0xEB7F96, 0x3AD26D, 0xE6CC76, 0x878751, 0xEAED38, 0xE27B4C, 0xB664C3, 0xC8E5CB, 0xA7221B, 0x418F66, 0xA8CAEF, 0x87658F, 0x561513, 0x31EAAE, 0x4B1235, 0x9C8516, 0x425AAF, 0xA16E51, 0x2C9290, 0xD6296D, 0xE2304B, 0xACE71F, 0xEC7C1F, 0x4B89A9, 0x1E2E3A, 0xECAB51, 0x5477EB, 0x53654B, 0xE9DDEA, 0xECE9B0, 0x89EBC2, 0x3C5872, 0xC25D52, 0xEB69E2, 0xC9E86A, 0xE970C0, 0xB28494, 0xAFAA8A, 0x659D53, 0x8E74BB, 0x449E26, 0xDEA870, 0xAC5A7A, 0x8DBD98, 0x6A439B, 0x336161, 0x7C5451, 0x54ECE0, 0x992F94, 0xA42875, 0x58400A, 0xE72BC9, 0x778DE6, 0xF0589E, 0xAA4E27, 0x7AB3BD, 0xB9AFB6, 0x2F1E3F, 0x8667CF, 0x2F410A, 0x49C393, 0x85394C, 0xA348C4, 0x4B7518, 0xE985B6, 0xD3E18D, 0x89EA53, 0x84972B, 0x43B162, 0x554666, 0xB266A8, 0x1C2E1C, 0xCC58EF, 0xAD6463, 0xCC5121, 0xEB6576, 0xBB9BEA, 0xB6BC32, 0xA795C9, 0x54BAA9, 0xC26F1E, 0x781E4E, 0xEA5C52, 0x4C4824, 0xF0ACAD, 0x88B955, 0x4E514C, 0x5FABD7, 0x903C35, 0xD02C1B, 0x817224, 0xC47EEE, 0x45CB2E, 0xA92D61, 0xAE7439, 0x5E8250, 0x5B1228, 0x552A60, 0xE98675, 0xE38CE7, 0xEAB1C9, 0x84E975, 0x381D2C, 0xAA8873, 0x68EF97, 0x7DBD7D, 0x3D5E95, 0xEBD039, 0xB6E784, 0x9991AB, 0x592F12, 0xE79F20, 0x558C79, 0x9CAB6C, 0x2C2809, 0x3C1918, 0x816D6F, 0x85CA2B, 0xB9CE97, 0x6A3A5C, 0x506628, 0x95E3D0, 0x338435, 0x92AEA6, 0xCA5287, 0xD28DD0, 0xC3A72B, 0xAD7A16, 0x5E7FC5, 0xB57BA6, 0x275639, 0x734633, 0xC14A9F, 0x98AAE0, 0x94E3EF, 0xC8395A, 0x7B745D, 0x876176, 0xDCC3E9, 0xAA2532, 0xE6D3CB, 0x842C0F, 0x3C3271, 0xBDAB61, 0xE95884, 0x90417E, 0x6F652F, 0x751520, 0x5785B8, 0xDED9BB, 0xE5E570, 0xD6C558, 0xCC3DBE, 0xBDE3DF, 0xE89037, 0x47D557, 0xC25E6D, 0x75739B, 0x5E3C45, 0x869044, 0xD88891, 0xC89A96, 0xAD9540, 0x5CC5EA, 0xDFCC93, 0xDA946E, 0xEC2C85, 0x835325, 0xAABAD1, 0xC9DE3B, 0xC4893A, 0x7C558F, 0xC2A872, 0xBCD8EA, 0xE797B9, 0x72EBAD, 0xB0EE48, 0xF3392E, 0x418D98, 0xADE564, 0x918860)
  def getUniqueColors(amount: Int): List[Int] = { //TODO remove this is not working
    val lowerLimit = 0x10;
    val upperLimit = 0xE0;
    val colorStep = ((upperLimit - lowerLimit) / Math.pow(amount, 1f / 3)).asInstanceOf[Int];

    val colors = new ListBuffer[Int]();

    var R = lowerLimit.asInstanceOf[Int]
    var G = lowerLimit.asInstanceOf[Int]
    var B = lowerLimit.asInstanceOf[Int]
    while (R < upperLimit) {
      while (G < upperLimit) {
        while (B < upperLimit) {
          if (colors.length >= amount) { //The calculated step is not very precise, so this safeguard is appropriate
            return colors.toList;
          } else {
            val color = (R << 16) + (G << 8) + (B);
            colors += (color);
          }
          B += colorStep
        }
        G += colorStep
      }
      R += colorStep
    }
    return colors.toList;
  }

  def reConstruct3dMat(labels: GraphLabels, dataLink: HashMap[Int, (Int, Int, Int)], xDim: Int, yDim: Int, zDim: Int): D3ArrInt = {

    val out = Array.fill(xDim, yDim, zDim) { 0 }
    for (i <- 0 until labels.d.size) {
      val coords = dataLink.get(i).get
      out(coords._1)(coords._2)(coords._3) = labels.d(i)
    }
    out
  }

  def printBMPFromGraph(graph: GraphStruct[Vector[Double], (Int, Int, Int)], labels: GraphLabels, slice3dAt: Int = 0, name: String = "non", colorMap: Map[Int, (Int, Int, Int)] = null) {

    val mask = readObjectFromFile[Array[Array[Array[Int]]]](graph.originMapFile)
    //Need to construct a new image.  
    val xDim = mask.length
    val yDim = mask(0).length

    val img: BufferedImage = new BufferedImage(xDim, yDim,
      BufferedImage.TYPE_INT_RGB);

    for (x <- 0 until xDim; y <- 0 until yDim) {
      val sID = mask(x)(y)(slice3dAt)
      val myClassPred = labels.d(sID)
      val myCol = if (colorMap == null) someColors(myClassPred % someColors.length) else {
        try {
          val rgb = colorMap.get(myClassPred).get
          new Color(rgb._1, rgb._2, rgb._3).getRGB()
        } catch {
          case e: Exception =>
            { println("Error myClassPred:" + myClassPred + " was not found") }
            return ()
        }
      }
      img.setRGB(x, y, myCol)

    }
    ImageIO.write(img, "BMP", new File("../data/debug/" + name + ".bmp")); //TODO change this output location
  }

  
  def  printSupPixLabels_Int(graph: GraphStruct[Vector[Double], (Int, Int, Int)], labels: GraphLabels, slice3dAt: Int = 0, name: String = "non", colorMap: Map[Int, Int] = null) {
    
    
    val mask = readObjectFromFile[Array[Array[Array[Int]]]](graph.originMapFile)
    //Need to construct a new image.  
    val xDim = mask.length
    val yDim = mask(0).length

    val old: BufferedImage = new BufferedImage(xDim, yDim,
      BufferedImage.TYPE_INT_RGB);

    
    val firstPixPerID = new HashMap [Int, (Int,Int)]
    
    for (x <- 0 until xDim; y <- 0 until yDim) {
      val sID = mask(x)(y)(slice3dAt)
      if(!firstPixPerID.contains(sID))
        firstPixPerID.put(sID,(x,y))
      val myClassPred = labels.d(sID)
      val myCol = if (colorMap == null) someColors(myClassPred % someColors.length) else {
        try {
          colorMap.get(myClassPred).get
        } catch {
          case e: Exception =>
            { println("Error myClassPred:" + myClassPred + " was not found") }
            return ()
        }
      }
      old.setRGB(x, y, myCol)

    }
    
    
    val  w = old.getWidth();
        val  h = old.getHeight();
        val img = new BufferedImage(w, h,BufferedImage.TYPE_INT_RGB);
        
     val g2d:Graphics2D = img.createGraphics();
        g2d.drawImage(old, 0, 0, null);
        g2d.setPaint(Color.black);
        g2d.setFont(new Font("Serif", Font.PLAIN, 8));
        
        
    firstPixPerID.foreach( a => {
      val id = a._1
      val x = a._2._1
      val y = a._2._2
      val s = ""+id;
      g2d.drawString(s, x, y);
    })
       g2d.dispose();
    
val pw = new PrintWriter(new File("../data/graph/" + name + "sID_graph_.txt"))

    for ( i <- 0 until graph.size){
      val id = i
      val neigh = graph.getC(id)
      
      pw.write("\n("+id+"-> ["+neigh.toList.mkString(",")+"] ")
    }
         ImageIO.write(img, "BMP", new File("../data/graph/" + name + "sID_.bmp")); //TODO change this output location
        
        pw.close 
   
  }
  
   def printBMPFromGraphInt(graph: GraphStruct[Vector[Double], (Int, Int, Int)], labels: GraphLabels, slice3dAt: Int = 0, name: String = "non", colorMap: Map[Int, Int] = null) {

    val mask = readObjectFromFile[Array[Array[Array[Int]]]](graph.originMapFile)
    //Need to construct a new image.  
    val xDim = mask.length
    val yDim = mask(0).length

    val img: BufferedImage = new BufferedImage(xDim, yDim,
      BufferedImage.TYPE_INT_RGB);

    for (x <- 0 until xDim; y <- 0 until yDim) {
      val sID = mask(x)(y)(slice3dAt)
      val myClassPred = labels.d(sID)
      val myCol = if (colorMap == null) someColors(myClassPred % someColors.length) else {
        try {
          colorMap.get(myClassPred).get
        } catch {
          case e: Exception =>
            { println("Error myClassPred:" + myClassPred + " was not found") }
            return ()
        }
      }
      img.setRGB(x, y, myCol)

    }
    ImageIO.write(img, "BMP", new File("../data/debug/" + name + ".bmp")); //TODO change this output location
  }

   
  //TODO find a nice scala-esq way of doing this. essencially i want the matlab function 'squeeze'
  def flatten3rdDim(in: D3ArrInt): Array[Array[Int]] = {

    val xDim = in.length
    val yDim = in(0).length
    val out = Array.fill(xDim, yDim) { 0 }
    for (x <- 0 until in.length) {
      for (y <- 0 until in(0).length) {
        out(x)(y) = in(x)(y)(0)
      }
    }
    return out

  }
  def printBMPfrom3dMat(in: Array[Array[Int]], fileName: String) = {

    val xDim = in.length
    val yDim = in(0).length
    val img: BufferedImage = new BufferedImage(xDim, yDim,
      BufferedImage.TYPE_INT_RGB);
    for (x <- 0 until xDim) {
      for (y <- 0 until yDim) {
        val tmp = ImageSegmentationUtils.colormapRev
        val toGet = in(x)(y)
        val tmpRGB = ImageSegmentationUtils.colormapRev.get(toGet)
        if (tmpRGB.isEmpty) {
          print("oops")
        }
        val rgb = tmpRGB.get
        img.setRGB(x, y, rgb)
      }
    }

    //TODO change the path. We dont want to assume things about the file structure below the pwd 
    ImageIO.write(img, "BMP", new File("../data/debug/" + fileName));

  }

  def nodeCmp(e1: Node[_], e2: Node[_]) = (e1.idx < e2.idx)

  def d3randomVecDlb(): D3ArrDbl = {
    Array.fill[Double](10, 10, 10) { Math.random() }

  }
  def d3randomVecInt(x: Int = 10, y: Int = 10, z: Int = 10): D3ArrInt = {
    Array.fill(x, y, z) { (Math.random() * 255).asInstanceOf[Int] }
  }

  def writeObjectToFile(filename: String, obj: AnyRef) = { // obj must have serialiazable trait
    import java.io._
    val fos = new FileOutputStream(filename)
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(obj)
    oos.close()
  }

  def readObjectFromFile[T](filename: String): T = {
    import java.io._
    val fis = new FileInputStream(filename)
    val ois = new ObjectInputStream(fis)
    val obj = ois.readObject()
    ois.close()
    obj.asInstanceOf[T]
  }

  def lossPerPixel(superPixelMapping: String, originalTrue: String, prediction: GraphLabels, lossFn: (Int, Int) => Double = (a: Int, b: Int) => { if (a == b) 0.0 else 1.0 }, colorMap: Map[(Int, Int, Int), Int]): Double = {
    assert(!originalTrue.equals("None"))
    val supPixToOrig = readObjectFromFile[Array[Array[Array[Int]]]](superPixelMapping)

    val openerGT = new Opener();
    val imgGT = openerGT.openImage(originalTrue);
    val gtStack = imgGT.getStack
    val gtColMod = gtStack.getColorModel

    val xDim = supPixToOrig.length
    val yDim = supPixToOrig(0).length
    val zDim = supPixToOrig(0)(0).length
    assert(xDim == gtStack.getWidth)
    assert(yDim == gtStack.getHeight)
    assert(zDim == gtStack.getSize)

    val eachLoss = for (x <- 0 until xDim; y <- 0 until yDim; z <- 0 until zDim) yield {
      val trueLabelColor = gtStack.getVoxel(x, y, z)
      val trueRGB = (gtColMod.getRed(trueLabelColor.asInstanceOf[Int]), gtColMod.getGreen(trueLabelColor.asInstanceOf[Int]), gtColMod.getBlue(trueLabelColor.asInstanceOf[Int]))
      val trueLabel = colorMap.get(trueRGB).get
      val requiredSuperPixelID = supPixToOrig(x)(y)(z)
      val predictedLabel = prediction.d(requiredSuperPixelID)
      lossFn(trueLabel, predictedLabel)
    }
    return (eachLoss.sum / eachLoss.length)
  }

  //Colormap should be (int,int) (color found in ground truth image, internal label represetnation)
   def lossPerPixelInt(superPixelMapping: String, originalTrue: String, prediction: GraphLabels, lossFn: (Int, Int) => Double = (a: Int, b: Int) => { if (a == b) 0.0 else 1.0 }, colorMap: Map[Int, Int]): Double = {
    assert(!originalTrue.equals("None"))
    val supPixToOrig = readObjectFromFile[Array[Array[Array[Int]]]](superPixelMapping)

    val openerGT = new Opener();
    val imgGT = openerGT.openImage(originalTrue);
    val gtStack = imgGT.getStack
    val gtColMod = gtStack.getColorModel

    val xDim = supPixToOrig.length
    val yDim = supPixToOrig(0).length
    val zDim = supPixToOrig(0)(0).length
    assert(xDim == gtStack.getWidth)
    assert(yDim == gtStack.getHeight)
    assert(zDim == gtStack.getSize)

    val eachLoss = for (x <- 0 until xDim; y <- 0 until yDim; z <- 0 until zDim) yield {
      val trueLabelColor = gtStack.getVoxel(x, y, z).asInstanceOf[Int]
      val trueLabel = colorMap.get(trueLabelColor).get
      val requiredSuperPixelID = supPixToOrig(x)(y)(z)
      val predictedLabel = prediction.d(requiredSuperPixelID)
      lossFn(trueLabel, predictedLabel)
    }
    return (eachLoss.sum / eachLoss.length)
  }
   
  val convertMSRC_counter = new AtomicInteger(0)
  def convertOT_msrc_toGraph(xi: DenseMatrix[ROIFeature], yi: DenseMatrix[ROILabel], numClasses: Int): (GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels) = {

    val nodeList = new scala.collection.mutable.ListBuffer[Node[Vector[Double]]]
    val labelVect = Array.fill(xi.rows * xi.cols)(0)
    val linkCoord = new HashMap[Int, (Int, Int, Int)]()
    val coordNode = new HashMap[(Int, Int, Int), Int]()
    val linkCoord_v2 = Array.fill(xi.rows, xi.cols, 1) { -1 }
    for (rIdx <- 0 until xi.rows) {
      for (cIdx <- 0 until xi.cols) {
        val myIDX = ImageSegmentationDemo.columnMajorIdx(rIdx, cIdx, xi.rows)
        val nextNode = new Node[Vector[Double]](myIDX, xi(rIdx, cIdx).feature, new HashSet[Int]())
        linkCoord.put(myIDX, (rIdx, cIdx, 0))
        linkCoord_v2(rIdx)(cIdx)(0) = myIDX
        coordNode.put((rIdx, cIdx, 0), nextNode.idx)
        labelVect(myIDX) = yi(rIdx, cIdx).label
        nodeList += nextNode
      }
    }
    val sortedNodeList = nodeList.sortWith(nodeCmp) //TODO make sure this is increasing order from 
    assert(sortedNodeList.length == xi.rows * xi.cols)
    assert(sortedNodeList(0).idx == 0 && sortedNodeList(1).idx == 1 && sortedNodeList(2).idx == 2)
    val nodeVect = Vector(sortedNodeList.toArray)

    linkCoord.keySet.foreach { key =>
      {
        val coords: (Int, Int, Int) = linkCoord.get(key).get
        nodeVect(key).connections ++= coordNode.get((coords._1 + 1, coords._2, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2 + 1, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2, coords._3 + 1))
        nodeVect(key).connections ++= coordNode.get((coords._1 - 1, coords._2, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2 - 1, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2, coords._3 - 1))
      }
    }

    //(new GraphStruct(nodeVect,linkCoord,(xi.rows-1,xi.cols-1,0)), new GraphLabels(Vector(labelVect),numClasses))
    val callID = convertMSRC_counter.getAndIncrement
    val maskDiskPath = "../data/" + "__msrcConvert_" + "__img_" + callID + ".mask" //TODO if this is taking too much space just remove the callTime so it overrides the last
    writeObjectToFile(maskDiskPath, linkCoord_v2)
    (new GraphStruct(nodeVect, maskDiskPath), new GraphLabels(Vector(labelVect), numClasses)) //TODO save raw labels in a file
  }

  def cutSuperPix(dataIn: D3ArrInt,
                  x: Int, y: Int, z: Int,
                  superPixSize: Vector[Int]): Array[Array[Array[Int]]] = {

    var out = new Array[Array[Array[Int]]](superPixSize(0))
    for {
      xIdx <- x until (x + superPixSize(0)); yIdx <- y until (y + superPixSize(1)); zIdx <- z until (z + superPixSize(2))
    } {
      val superX = xIdx - x
      val superY = yIdx - y
      val superZ = zIdx - z
      if (out(superX) == null)
        out(superX) = new Array[Array[Int]](superPixSize(2))
      if (out(superX)(superY) == null)
        out(superX)(superY) = new Array[Int](superPixSize(3))

      out(superX)(superY)(superZ) = dataIn(x)(y)(z)
    }
    out
  }

  def simple3dhist(in: Array[Array[Array[Int]]], numBins: Int, histBinSize: Int): Vector[Double] = {
    val out = Vector.fill(numBins)(0.0)
    in.view.flatten.flatten.foreach(dat => out(Math.floor((dat - 0.000001) / histBinSize).asInstanceOf[Int]) += 1)
    out
  }

  def graphFrom3dMat(dataIn: Array[Array[Array[Int]]],
                     labels: Array[Array[Array[Int]]],
                     featureFn: Array[Array[Array[Int]]] => Vector[Double],
                     inSuperPixelDim: Vector[Int]): (GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels) = {

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
      var out = Array.fill(superPixSize(0), superPixSize(1), superPixSize(2)) { 0 }
      for {
        xIdx <- x until (x + superPixSize(0)); yIdx <- y until (y + superPixSize(1)); zIdx <- z until (z + superPixSize(2))
      } {
        val superX = xIdx - x
        val superY = yIdx - y
        val superZ = zIdx - z

        if (x >= dataIn.length || y >= dataIn(0).length || z >= dataIn(0)(0).length)
          return null
        out(superX)(superY)(superZ) = dataIn(x)(y)(z)

      }
      out
    }

    val nodeList = new scala.collection.mutable.ListBuffer[Node[Vector[Double]]]
    var counter = 0;
    //val coordLink = new HashMap[(Int,Int,Int),Int]()
    val linkCoord = new HashMap[Int, (Int, Int, Int)]()
    val coordNode = new HashMap[(Int, Int, Int), Int]()
    val labelOut = new LinkedList[Int]()
    val linkCoord_v2 = Array.fill(xDim, yDim, zDim) { -1 }
    for {
      supX <- 0 until numSupPixelPerX; supY <- 0 until numSupPixelPerY; supZ <- 0 until numSupPixelPerZ
    } {

      val supPixData = myCutSuperPix(supX * superPixSize(0), supY * superPixSize(1), supZ * superPixSize(2))
      val feats = featureFn(supPixData)
      val totalEleNum = superPixSize.reduceLeft((a1, a2) => a1 * a2).toDouble
      val supPixSum = supPixData.flatMap { x => x }.flatMap { x => x }.toList.reduceLeft((a1, a2) => a1 + a2) / totalEleNum
      labelOut.append(LinkedList(if (supPixSum > 0.5) 1 else 0)) //TODO this should be the most freq occuring label. So it generalizes to more than binary 
      //TODO there is a bug in this append statment
      val nextNode = new Node[Vector[Double]](counter, feats, new HashSet[Int]())
      linkCoord.put(counter, (supX, supY, supZ)) //TODO remove 

      for (
        linkX <- supX * superPixSize(0) until ((supX + 1) * superPixSize(0));
        linkY <- supY * superPixSize(1) until ((supY + 1) * superPixSize(1));
        linkZ <- supZ * superPixSize(2) until ((supZ + 1) * superPixSize(2))
      ) {
        linkCoord_v2(linkX)(linkY)(linkZ) = counter
      }

      coordNode.put((supX, supY, supZ), nextNode.idx)
      nodeList += nextNode
      counter += 1
    }

    val outLabelGraph = new GraphLabels(Vector(labelOut.toArray), 2) //TODO save raw labels in a file
    val nodeVect = Vector(nodeList.toArray)

    linkCoord.keySet.foreach { key =>
      {
        val coords = linkCoord.get(key).get
        nodeVect(key).connections ++= coordNode.get((coords._1 + 1, coords._2, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2 + 1, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2, coords._3 + 1))
        nodeVect(key).connections ++= coordNode.get((coords._1 - 1, coords._2, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2 - 1, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2, coords._3 - 1))

      }
    }

    val callID = convertMSRC_counter.getAndIncrement
    val maskDiskPath = "../data/" + "__3dGen_" + "__img_" + callID + ".mask" //TODO if this is taking too much space just remove the callTime so it overrides the last
    writeObjectToFile(maskDiskPath, linkCoord_v2)

    // (new GraphStruct[Vector[Double],(Int,Int,Int)](nodeVect,linkCoord,(xDim-1,yDim-1,zDim-1)), outLabelGraph)
    (new GraphStruct[Vector[Double], (Int, Int, Int)](nodeVect, maskDiskPath), outLabelGraph)
  }

  def boundGet[cont](x: Int, y: Int, z: Int, data: Array[Array[Array[cont]]]): Option[cont] = {
    if (x >= 0 & y >= 0 && z >= 0 && x < data.length && y < data(0).length && z < data(0)(0).length) {
      return Option(data(x)(y)(z))
    } else
      return None
  }

  //THIS DId not workout how i wanted it 
  def genClusteredGraphData(canvasSize: Int, probUnifRandom: Double, featureNoise: Double, pairRandomItr: Int, numClasses: Int, neighbouringProb: Array[Array[Double]]) {
    var unaryData = Array.fill(canvasSize, canvasSize, 1)(0)
    for (x <- 0 until unaryData.length) {
      for (y <- 0 until unaryData(0).length) {
        for (z <- 0 until unaryData(0)(0).length) {
          unaryData(x)(y)(z) = (Math.random() * numClasses.toDouble).toInt % numClasses

        }
      }
    }

    var pairData = Array.fill(canvasSize, canvasSize, 1)(0)
    for (idx <- 0 until pairRandomItr) {
      for (x <- 0 until pairData.length) {
        for (y <- 0 until pairData(0).length) {
          for (z <- 0 until pairData(0)(0).length) {

            val neigh = new LinkedList[Vector[Double]]()
            //val self = theData(x)(y)(z)
            var totalProbsPerNeighbouringLabel = Vector(Array.fill(numClasses)(0.0))
            var xN = x; var yN = y; var zN = z
            if (xN >= 0 && yN >= 0 && zN >= 0 && xN < pairData.length && yN < pairData(0).length && zN < pairData(0)(0).length) {
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel += Vector(neighbouringProb(other))
            }
            xN = x + 1; yN = y; zN = z
            if (xN >= 0 && yN >= 0 && zN >= 0 && xN < pairData.length && yN < pairData(0).length && zN < pairData(0)(0).length) {
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel += Vector(neighbouringProb(other))
            }
            xN = x; yN = y + 1; zN = z
            if (xN >= 0 && yN >= 0 && zN >= 0 && xN < pairData.length && yN < pairData(0).length && zN < pairData(0)(0).length) {
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel += Vector(neighbouringProb(other))
            }
            xN = x; yN = y; zN = z + 1
            if (xN >= 0 && yN >= 0 && zN >= 0 && xN < pairData.length && yN < pairData(0).length && zN < pairData(0)(0).length) {
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel += Vector(neighbouringProb(other))
            }
            xN = x - 1; yN = y; zN = z
            if (xN >= 0 && yN >= 0 && zN >= 0 && xN < pairData.length && yN < pairData(0).length && zN < pairData(0)(0).length) {
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel += Vector(neighbouringProb(other))
            }
            xN = x; yN = y - 1; zN = z
            if (xN >= 0 && yN >= 0 && zN >= 0 && xN < pairData.length && yN < pairData(0).length && zN < pairData(0)(0).length) {
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel += Vector(neighbouringProb(other))
            }
            xN = x; yN = y; zN = z - 1
            if (xN >= 0 && yN >= 0 && zN >= 0 && xN < pairData.length && yN < pairData(0).length && zN < pairData(0)(0).length) {
              val other = unaryData(xN)(yN)(zN)
              totalProbsPerNeighbouringLabel += Vector(neighbouringProb(other))
            }

            pairData(x)(y)(z) = randomLabelWithProb(totalProbsPerNeighbouringLabel.toArray)
          }
        }
      }

      unaryData = pairData
      //  println("-------------------------------------------------")

    }
    println(unaryData.deep.mkString("\n"))
    print("tst")
  }

  def randomLabelWithProb(probs: Array[Double]): Int = {
    val totalProbs = probs.toList.reduceLeft((a1, a2) => a1 + a2)
    var randomVal = Math.random() * totalProbs
    for (idx <- 0 until probs.length) { //this can purhaps be done faster with a map or something 

      randomVal -= probs(idx)
      if (randomVal <= 0)
        return idx
    }
    return probs.length - 1
  }

  def anotherDataGenFn(canvasSize: Int, portionBackground: Double, numClasses: Int, featureNoise: Double, random: java.util.Random): (GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels) = {
    //Gen random mat of 10x10 
    //Force it to be mostly zeros by rounding down after 0.8
    //Per 1 that still exists choose one of the non zero classes to replace it with
    //Generate a much larger 100x100 where each pixel of original mat represents 10x10 pixes of that value
    //Add Random noise 
    //Generate a graph with excisting functions and make the superpixel size 5x5 
    assert(canvasSize % 4 == 0)

    val topLvl = Array.fill(canvasSize / 4, canvasSize / 4) { if (random.nextDouble() < portionBackground) 0 else 1 }
    val scaled = DenseMatrix.zeros[Int](canvasSize, canvasSize)
    val maskCacheFormat = Array.fill(canvasSize, canvasSize, 1) { -1 }

    //Lets go directly to a graph
    for (x <- 0 until canvasSize / 4) {
      for (y <- 0 until canvasSize / 4) {
        if (topLvl(x)(y) != 0) {
          val curClass = random.nextInt(numClasses - 1) + 1
          scaled(x * 4 to x * 4 + 3, y * 4 to y * 4 + 3) := curClass

          for (lX <- x * 4 to x * 4 + 3; lY <- y * 4 to y * 4 + 3) {
            maskCacheFormat(lX)(lY)(0) = curClass
          }

        }
      }
    }

    def featureFn(label: Int): Vector[Double] = {
      val outF: Array[Double] = if (random.nextDouble() > featureNoise) {
        val tmp = Array.fill[Double](numClasses) { random.nextDouble() }
        tmp(label) = 1.0
        tmp
      } else {
        Array.fill(numClasses) { random.nextDouble() }
      }
      normalize(Vector(outF))
    }

    val nodeList = new scala.collection.mutable.ListBuffer[Node[Vector[Double]]]
    var counter = 0;
    //val coordLink = new HashMap[(Int,Int,Int),Int]()
    val linkCoord = new HashMap[Int, (Int, Int, Int)]()
    val coordNode = new HashMap[(Int, Int, Int), Int]()
    val labelOut = new ListBuffer[Int]
    for {
      supX <- 0 until canvasSize; supY <- 0 until canvasSize
    } {

      val feats = featureFn(scaled(supX, supY))

      labelOut ++= LinkedList(scaled(supX, supY))
      val nextNode = new Node[Vector[Double]](counter, feats, new HashSet[Int]())
      linkCoord.put(counter, (supX, supY, 0))

      coordNode.put((supX, supY, 0), nextNode.idx)
      nodeList += nextNode
      counter += 1
    }

    val outLabelGraph = new GraphLabels(Vector(labelOut.toArray), numClasses) //TODO save raw labels in file 
    val nodeVect = Vector(nodeList.toArray)

    linkCoord.keySet.foreach { key =>
      {
        val coords = linkCoord.get(key).get
        nodeVect(key).connections ++= coordNode.get((coords._1 + 1, coords._2, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2 + 1, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2, coords._3 + 1))
        nodeVect(key).connections ++= coordNode.get((coords._1 - 1, coords._2, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2 - 1, coords._3))
        nodeVect(key).connections ++= coordNode.get((coords._1, coords._2, coords._3 - 1))

      }
    }

    val callID = convertMSRC_counter.getAndIncrement
    val maskDiskPath = "../data/" + "__3dGen_" + "__img_" + callID + ".mask" //TODO if this is taking too much space just remove the callTime so it overrides the last
    writeObjectToFile(maskDiskPath, maskCacheFormat)

    //(new GraphStruct[Vector[Double],(Int,Int,Int)](nodeVect,linkCoord,(canvasSize-1,canvasSize-1,0)), outLabelGraph)
    (new GraphStruct[Vector[Double], (Int, Int, Int)](nodeVect, maskDiskPath), outLabelGraph)

  }

  def genSquareBlobs(howMany: Int, anvasSize: Int, portionBackground: Double, numClasses: Int, featureNoise: Double, randSeed: Int): Seq[LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels]] = {
    val random = if (randSeed != (-1)) new java.util.Random(randSeed) else new java.util.Random()
    val out = for (i <- 0 until howMany) yield {

      val (xGraph, yList) = anotherDataGenFn(anvasSize, portionBackground, numClasses, featureNoise, random)
      new LabeledObject[GraphStruct[breeze.linalg.Vector[Double], (Int, Int, Int)], GraphLabels](yList, xGraph)
    }
    out
  }

  def genColorfullSquaresData(howMany: Int, canvasSize: Int, squareSize: Int, portionBackground: Double, numClasses: Int, featureNoise: Double, outputDir: String, randomSeed:Int=(-1) ){
    assert(canvasSize % squareSize == 0)
    assert(portionBackground <= 1)
    assert(featureNoise <= 1)
    val random = if(randomSeed==(-1)) new java.util.Random() else new java.util.Random(randomSeed)
     val colorMap = Random.shuffle(someColorsRGB)
    val pathF = new File(outputDir + "/Images")
    val pathFGT = new File(outputDir + "/GroundTruth")
    if (!pathF.exists())
      pathF.mkdirs()
    if (!pathFGT.exists())
      pathFGT.mkdirs()

    for (count <- 1 to howMany) {
      val uberPix = Array.fill(canvasSize / squareSize, canvasSize / squareSize) { -1 }
      val outLabel = Array.fill(canvasSize, canvasSize) { -1 }

      for (uX <- 0 until uberPix.length; uY <- 0 until uberPix(0).length) {
        if (random.nextDouble() < portionBackground)
          uberPix(uX)(uY) = 0
        else
          uberPix(uX)(uY) = random.nextInt(numClasses - 1) + 1
      }
      for (x <- 0 until canvasSize; y <- 0 until canvasSize) {
        outLabel(x)(y) = uberPix(x / squareSize)(y / squareSize)
      }

      val imgData: BufferedImage = new BufferedImage(canvasSize, canvasSize,
        BufferedImage.TYPE_INT_RGB);
      val imgGT: BufferedImage = new BufferedImage(canvasSize, canvasSize,
        BufferedImage.TYPE_INT_RGB);

     

      for (x <- 0 until canvasSize; y <- 0 until canvasSize) {

        val myLabel = outLabel(x)(y)
        val rN = min(255, max(0, (((1 - featureNoise) * colorMap(myLabel)._1 + featureNoise * random.nextInt(256)) / 2).asInstanceOf[Int]))
        val gN = min(255, max(0, (((1 - featureNoise) * colorMap(myLabel)._2 + featureNoise * random.nextInt(256)) / 2).asInstanceOf[Int]))
        val bN = min(255, max(0, (((1 - featureNoise) * colorMap(myLabel)._3 + featureNoise * random.nextInt(256)) / 2).asInstanceOf[Int]))

        val noisyColor = new Color(rN, gN, bN).getRGB
        imgData.setRGB(x, y, noisyColor)
        val trueColor = new Color(colorMap(myLabel)._1, colorMap(myLabel)._2, colorMap(myLabel)._3).getRGB
        imgGT.setRGB(x, y, trueColor)

      }
      ImageIO.write(imgData, "BMP", new File(outputDir + "/Images/genImg_" + count + ".bmp")); //TODO change this output location
      ImageIO.write(imgGT, "BMP", new File(outputDir + "/GroundTruth/genImg_" + count + "_GT.bmp")); //TODO change this output location
    }
  
  }
  
  
  def genColorfullSquaresDataSuperNoise(howMany: Int, canvasSize: Int, squareSize: Int, portionBackground: Double, numClasses: Int, featureNoise: Double,fineGrainNoise:Double=0.0, outputDir: String, randomSeed:Int=(-1) ){
    assert(canvasSize % squareSize == 0)
    assert(portionBackground <= 1)
    assert(featureNoise <= 1)
    val random = if(randomSeed==(-1)) new java.util.Random() else new java.util.Random(randomSeed)
     val colorMap = Random.shuffle(someColorsRGB)
    val pathF = new File(outputDir + "/Images")
    val pathFGT = new File(outputDir + "/GroundTruth")
    if (!pathF.exists())
      pathF.mkdirs()
    if (!pathFGT.exists())
      pathFGT.mkdirs()

    for (count <- 1 to howMany) {
      val uberPix = Array.fill(canvasSize / squareSize, canvasSize / squareSize) { -1 }
      val uberNoise =Array.fill(canvasSize / squareSize, canvasSize / squareSize) { (-1,-1,-1) }
      val outLabel = Array.fill(canvasSize, canvasSize) { -1 }

      
       for (uX <- 0 until uberPix.length; uY <- 0 until uberPix(0).length) {
        uberNoise(uX)(uY)=(random.nextInt(256),random.nextInt(256),random.nextInt(256))
      }
       
       
      for (uX <- 0 until uberPix.length; uY <- 0 until uberPix(0).length) {
        if (random.nextDouble() < portionBackground)
          uberPix(uX)(uY) = 0
        else
          uberPix(uX)(uY) = random.nextInt(numClasses - 1) + 1
      }
      for (x <- 0 until canvasSize; y <- 0 until canvasSize) {
        outLabel(x)(y) = uberPix(x / squareSize)(y / squareSize)
      }

      val imgData: BufferedImage = new BufferedImage(canvasSize, canvasSize,
        BufferedImage.TYPE_INT_RGB);
      val imgGT: BufferedImage = new BufferedImage(canvasSize, canvasSize,
        BufferedImage.TYPE_INT_RGB);

      
     

      for (x <- 0 until canvasSize; y <- 0 until canvasSize) {

        val myLabel = outLabel(x)(y)
        val uX = x/squareSize
        val uY = y/squareSize
        val rN =  min(255, max(0, (( (1-fineGrainNoise)*(((1 - featureNoise) * colorMap(myLabel)._1 + featureNoise * uberNoise(uX)(uY)._1) )+ fineGrainNoise*random.nextInt(255) )   ).asInstanceOf[Int]))
        val gN =  min(255, max(0, (( (1-fineGrainNoise)*(((1 - featureNoise) * colorMap(myLabel)._2 + featureNoise * uberNoise(uX)(uY)._2) )+ fineGrainNoise*random.nextInt(255) )   ).asInstanceOf[Int]))
        val bN =  min(255, max(0, (( (1-fineGrainNoise)*(((1 - featureNoise) * colorMap(myLabel)._3 + featureNoise * uberNoise(uX)(uY)._3) )+ fineGrainNoise*random.nextInt(255) )   ).asInstanceOf[Int]))

        val noisyColor = new Color(rN, gN, bN).getRGB
        imgData.setRGB(x, y, noisyColor)
        val trueColor = new Color(colorMap(myLabel)._1, colorMap(myLabel)._2, colorMap(myLabel)._3).getRGB
        imgGT.setRGB(x, y, trueColor)

      }
      ImageIO.write(imgData, "BMP", new File(outputDir + "/Images/genImg_" + count + ".bmp")); //TODO change this output location
      ImageIO.write(imgGT, "BMP", new File(outputDir + "/GroundTruth/genImg_" + count + "_GT.bmp")); //TODO change this output location
    }
  
   val pw = new PrintWriter(new File(outputDir+"/dataGenerationConfig.cfg" ))
   
   pw.write("howMany:"+howMany)
       pw.write("\ncanvasSize:"+canvasSize)
       pw.write("\nsquareSize:"+squareSize) 
       pw.write("\nportionBackground:"+portionBackground) 
       pw.write("\nnumClasses:"+numClasses) 
       pw.write("\nfeatureNoise:"+featureNoise)
       pw.write("\nfineGrainNoise:"+fineGrainNoise) 
       pw.write("\noutputDir:"+outputDir) 
       pw.write("\nrandomSeed:"+randomSeed)
pw.write(" ")
pw.close
    
  }

    def genGreyfullSquaresDataSuperNoise(howMany: Int, canvasSize: Int, squareSize: Int, portionBackground: Double, numClasses: Int, featureNoise: Double,fineGrainNoise:Double=0.0, outputDir: String, randomSeed:Int=(-1) ){
    assert(canvasSize % squareSize == 0)
    assert(portionBackground <= 1)
    assert(featureNoise <= 1)
    val random = if(randomSeed==(-1)) new java.util.Random() else new java.util.Random(randomSeed)
     val colorMap = Random.shuffle(someColorsRGB)
    val pathF = new File(outputDir + "/Images")
    val pathFGT = new File(outputDir + "/GroundTruth")
    if (!pathF.exists())
      pathF.mkdirs()
    if (!pathFGT.exists())
      pathFGT.mkdirs()

    for (count <- 1 to howMany) {
      val uberPix = Array.fill(canvasSize / squareSize, canvasSize / squareSize) { -1 }
      val uberNoise =Array.fill(canvasSize / squareSize, canvasSize / squareSize) { (-1,-1,-1) }
      val outLabel = Array.fill(canvasSize, canvasSize) { -1 }

      
       for (uX <- 0 until uberPix.length; uY <- 0 until uberPix(0).length) {
        uberNoise(uX)(uY)=(random.nextInt(256),random.nextInt(256),random.nextInt(256))
      }
       
       
      for (uX <- 0 until uberPix.length; uY <- 0 until uberPix(0).length) {
        if (random.nextDouble() < portionBackground)
          uberPix(uX)(uY) = 0
        else
          uberPix(uX)(uY) = random.nextInt(numClasses - 1) + 1
      }
      for (x <- 0 until canvasSize; y <- 0 until canvasSize) {
        outLabel(x)(y) = uberPix(x / squareSize)(y / squareSize)
      }

      val imgData: BufferedImage = new BufferedImage(canvasSize, canvasSize,
        BufferedImage.TYPE_BYTE_GRAY);
      val imgGT: BufferedImage = new BufferedImage(canvasSize, canvasSize,
        BufferedImage.TYPE_INT_RGB);

      
     

      for (x <- 0 until canvasSize; y <- 0 until canvasSize) {

        val myLabel = outLabel(x)(y)
        val uX = x/squareSize
        val uY = y/squareSize
        val rN =  min(255, max(0, (( (1-fineGrainNoise)*(((1 - featureNoise) * colorMap(myLabel)._1 + featureNoise * uberNoise(uX)(uY)._1) )+ fineGrainNoise*random.nextInt(255) )   ).asInstanceOf[Int]))
        val gN =  min(255, max(0, (( (1-fineGrainNoise)*(((1 - featureNoise) * colorMap(myLabel)._2 + featureNoise * uberNoise(uX)(uY)._2) )+ fineGrainNoise*random.nextInt(255) )   ).asInstanceOf[Int]))
        val bN =  min(255, max(0, (( (1-fineGrainNoise)*(((1 - featureNoise) * colorMap(myLabel)._3 + featureNoise * uberNoise(uX)(uY)._3) )+ fineGrainNoise*random.nextInt(255) )   ).asInstanceOf[Int]))

        val gray = (rN + gN +bN )/3
        val noisyColor = new Color(gray, gray, gray).getRGB
        imgData.setRGB(x, y, noisyColor)
        val trueColor = new Color(colorMap(myLabel)._1, colorMap(myLabel)._2, colorMap(myLabel)._3).getRGB
        imgGT.setRGB(x, y, trueColor)

      }
      ImageIO.write(imgData, "BMP", new File(outputDir + "/Images/genImg_" + count + ".bmp")); //TODO change this output location
      ImageIO.write(imgGT, "BMP", new File(outputDir + "/GroundTruth/genImg_" + count + "_GT.bmp")); //TODO change this output location
    }
  
   val pw = new PrintWriter(new File(outputDir+"/dataGenerationConfig.cfg" ))
   
   pw.write("isGray: True")
   pw.write("howMany:"+howMany)
       pw.write("\ncanvasSize:"+canvasSize)
       pw.write("\nsquareSize:"+squareSize) 
       pw.write("\nportionBackground:"+portionBackground) 
       pw.write("\nnumClasses:"+numClasses) 
       pw.write("\nfeatureNoise:"+featureNoise)
       pw.write("\nfineGrainNoise:"+fineGrainNoise) 
       pw.write("\noutputDir:"+outputDir) 
       pw.write("\nrandomSeed:"+randomSeed)
pw.write(" ")
pw.close
    
  }

  
}