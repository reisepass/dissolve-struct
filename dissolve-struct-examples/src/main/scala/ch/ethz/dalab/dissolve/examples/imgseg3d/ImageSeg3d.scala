package ch.ethz.dalab.dissolve.examples.imgseg3d

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Buffer
import scala.io.Source

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.normalize
import cc.factorie.infer.MaximizeByMPLP
import cc.factorie.infer.SamplingMaximizer
import cc.factorie.infer.VariableSettingsSampler
import cc.factorie.model.CombinedModel
import cc.factorie.model.Factor
import cc.factorie.model.Factor1
import cc.factorie.model.Factor2
import cc.factorie.model.ItemizedModel
import cc.factorie.model.TupleTemplateWithStatistics2
import cc.factorie.singleFactorIterable
import cc.factorie.variable.DiscreteDomain
import cc.factorie.variable.DiscreteVariable
import cc.factorie.variable.IntegerVariable
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.optimization.RoundLimitCriterion
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.SolverUtils

object ImageSeg3d extends DissolveFunctions[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]] with Serializable {
  val DISABLE_PAIRWISE: Boolean = false
  /*
   * Counts occurances of adjacent pairs of classes 
   * 
   * 
   */

  def columnMajorIdx3d(x: Int, y: Int, z: Int, dimX: Int, dimY: Int) = x + dimX * y + dimX * dimY * z

  /**
   * Given:
   * - a matrix xMat of super-pixels, of size r = n x m, and x_i, an f-dimensional vector
   * - corresponding labels of these super-pixels yMat, with K classes
   * Return:
   * - a matrix of size (f*K) x r, i.e, each column corresponds to the feature map of x_i
   */

  // We assume classes are given by sequencial Int values  STARTING AT 0 
  def getUnaryFeatureMapPadded(yMat: NominalThreeDimMat[Int], xMat: ThreeDimMat[Array[Double]]): DenseMatrix[Double] = {
    assert(xMat.xDim == yMat.xDim)
    assert(xMat.yDim == yMat.yDim)
    assert(xMat.zDim == yMat.zDim)

    val numFeatures = xMat.get(0, 0, 0).length //TODO this hist feature size uniformity is not garateed inside my datastructure 
    val numClasses = yMat.classSet.size

    assert(yMat.classSet.toSeq.sum == (0 until numClasses).reduceLeft[Int](_ + _)) //Testing sequencial assumoption rufly 

    val numRegions = xMat.xDim * xMat.yDim * xMat.zDim

    val unaryMat = DenseMatrix.zeros[Double](numFeatures * numClasses, numRegions)

    /**
     * Populate unary features
     * For each node i in graph defined by xMat, whose feature vector is x_i and corresponding label is y_i,
     * construct a feature map phi_i given by: [I(y_i = 0)x_i I(y_i = 1)x_i ... I(y_i = K)x_i ]
     */

    for (x <- 0 until xMat.xDim) {
      for (y <- 0 until xMat.yDim) {
        for (z <- 0 until xMat.zDim) {
          val i = columnMajorIdx3d(x, y, z, xMat.xDim, xMat.yDim)
          val x_i = xMat.get(x, y, z)
          val y_i = yMat.get(x, y, z)
          val phi_i = DenseVector.zeros[Double](numFeatures * numClasses)
          val startIdx = numFeatures * y_i //assumption of y_i being an int comes in here. Else this would have to have some static ordering 
          val endIdx = startIdx + numFeatures
          phi_i(startIdx until endIdx) := Vector(x_i)
          unaryMat(::, i) := phi_i

        }
      }
    }

    unaryMat
  }

  /**
   * Given:
   * - a matrix xMat of super-pixels, of size r = n x m, and x_i, an f-dimensional vector
   * - corresponding labels of these super-pixels yMat, with K classes
   * Return:
   * - a matrix of size f x r, each column corresponding to a (histogram) feature vector of region r
   */
  def getUnaryFeatureMap_justX(xMat: ThreeDimMat[Array[Double]]): DenseMatrix[Double] = {

    val numFeatures = xMat.get(0, 0, 0).length
    val numRegions = xMat.xDim * xMat.yDim * xMat.zDim

    val unaryMat = DenseMatrix.zeros[Double](numFeatures, numRegions)

    /**
     * Populate unary features
     * For each node i in graph defined by xMat, whose feature vector is x_i and corresponding label is y_i,
     * construct a feature map phi_i given by: [I(y_i = 0)x_i I(y_i = 1)x_i ... I(y_i = K)x_i ]
     */

    for (x <- 0 until xMat.xDim) {
      for (y <- 0 until xMat.yDim) {
        for (z <- 0 until xMat.zDim) {
          val i = columnMajorIdx3d(x, y, z, xMat.xDim, xMat.yDim)
          val x_i = xMat.get(x, y, z)
          unaryMat(::, i) := Vector(x_i)

        }
      }
    }
    unaryMat
  }

  def getPairwiseFeatureMap(yMat: NominalThreeDimMat[Int], xMat: ThreeDimMat[Array[Double]]): DenseMatrix[Double] = {

    assert(xMat.xDim == yMat.xDim)
    assert(xMat.yDim == yMat.yDim)
    assert(xMat.zDim == yMat.zDim)

    val numFeatures = xMat.get(0, 0, 0).length //TODO this hist feature size uniformity is not garateed inside my datastructure 
    val numClasses = yMat.classSet.size
    val numRegions = xMat.xDim * xMat.yDim * xMat.zDim

    val pairwiseMat = DenseMatrix.zeros[Double](numClasses, numClasses)

    for (
      y <- 0 until xMat.xDim;
      x <- 0 until xMat.yDim;
      z <- 0 until xMat.zDim
    ) {
      val classA = yMat.get(x, y, z)

      val neighbours = List((1, 0, 0), (0, 1, 0), (0, 0, 1))

      for ((dx, dy, dz) <- neighbours if (x + dx >= 0) && (y + dy >= 0) && (z + dz >= 0) && (x + dx < xMat.xDim) && (y + dy < xMat.yDim) && (z + dz < xMat.zDim)) {
        val classB = yMat.get(x + dx, y + dy, z + dz)
        pairwiseMat(classA, classB) += 1.0
        pairwiseMat(classB, classA) += 1.0
      }
    }

    pairwiseMat
  }

  /**
   * Feature Function.
   * Uses: http://arxiv.org/pdf/1408.6804v2.pdf
   * http://www.kev-smith.com/papers/LUCCHI_ECCV12.pdf
   */
  def featureFn(xMat: ThreeDimMat[Array[Double]], yMat: NominalThreeDimMat[Int]): Vector[Double] = {

    assert(xMat.xDim == yMat.xDim)
    assert(xMat.yDim == yMat.yDim)
    assert(xMat.zDim == yMat.zDim)
    if (!yMat.nominal)
      assert(yMat.nominal)

    val numDims = xMat.get(0, 0, 0).length //TODO this hist feature size uniformity is not garateed inside my datastructure 
    val numClasses = yMat.classSet.size

    val unaryFeatureSize = numDims * numClasses
    val pairwiseFeatureSize = numClasses * numClasses
    val phi = if (!DISABLE_PAIRWISE) DenseVector.zeros[Double](unaryFeatureSize + pairwiseFeatureSize) else DenseVector.zeros[Double](unaryFeatureSize)

    // Unaries vector, of size f * K. Each column corresponds to a feature vector
    val unary = DenseVector.zeros[Double](numDims * numClasses)

    for (
      y <- 0 until xMat.xDim;
      x <- 0 until xMat.yDim;
      z <- 0 until xMat.zDim
    ) { //TODO can't you just call  getUnaryFeatureMap()  for this ? 
      val label = yMat.get(x, y, z)
      val startIdx = label * numDims
      val endIdx = startIdx + numDims
      unary(startIdx until endIdx) := Vector(xMat.get(x, y, z)) + unary(startIdx until endIdx)
    }
    // Set Unary Features
    phi(0 until (numDims * numClasses)) := unary

    if (!DISABLE_PAIRWISE) {
      val pairwise = normalize(getPairwiseFeatureMap(yMat, xMat).toDenseVector) //TODO does this toDenseVector actually use proper columnIndex form ? 
      phi((numDims * numClasses) until phi.size) := pairwise
    }

    phi
  }

  /**
   * Loss function
   */
  def lossFn(yTruth: NominalThreeDimMat[Int], yPredict: NominalThreeDimMat[Int]): Double = {

    assert(yPredict.xDim == yTruth.xDim)
    assert(yPredict.yDim == yTruth.yDim)
    assert(yPredict.zDim == yTruth.zDim)
    val classFreqs = yTruth.classFreq

    val loss =
      for (
        x <- 0 until yTruth.xDim;
        y <- 0 until yTruth.yDim;
        z <- 0 until yTruth.zDim
      ) yield {
        //TODO put this back to weighted loss 
        //if (yTruth.get(x, y, z) == yPredict.get(x, y, z)) 0.0 else 1.0 / classFreqs.get(yTruth.get(x, y, z)).get // Insert classFrequency back into the truthObject
        if (yTruth.get(x, y, z) == yPredict.get(x, y, z)) 0.0 else 1.0 // Insert classFrequency back into the truthObject
      }

    loss.sum / (yTruth.xDim * yTruth.yDim * yTruth.zDim)
  }

  def oracleFn_MaxPerPixel(model: StructSVMModel[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]], xi: ThreeDimMat[Array[Double]], yi: NominalThreeDimMat[Int]): NominalThreeDimMat[Int] = {

    //TODO yi is the truth for this particular xi, since in this algo we update a random xi constraint block at a time 

    val numFeatures = xi.get(0, 0, 0).length
    val numLabels = yi.classSet.size
    val numLabelsModel = model.numClasses
    assert(model.weights.size == (numFeatures * yi.classSet.size))

    var out: NominalThreeDimMat[Int] = new NominalThreeDimMat(Vector(yi.xDim, yi.yDim, yi.zDim), classes = yi.classSet) //TODO here i put an error 

    for (x <- 0 until yi.xDim) {
      for (y <- 0 until yi.yDim) {
        for (z <- 0 until yi.zDim) {
          var maxCost = Double.MinValue
          var maxCostY = 0 //again assuming y_classes are consecutive integers
          for (possibleY <- yi.classSet.toSeq) {
            val x_i = xi.get(x, y, z)

            val phi_i = DenseVector.zeros[Double](numFeatures * yi.classSet.size)
            val startIdx = numFeatures * possibleY //assumption of y_i being an int comes in here. Else this would have to have some static ordering 
            val endIdx = startIdx + numFeatures

            phi_i(startIdx until endIdx) := Vector(x_i)

            val partialEnergy = model.weights dot phi_i

            val someYi = yi.get(x, y, z)
            val partialLoss = (if (someYi == possibleY)
              0.0
            else
              1.0 /
                yi.classFreq
                .get(someYi)
                .get) /
              (yi.xDim * yi.yDim * yi.zDim)

            if (maxCost < (partialLoss - partialEnergy)) maxCost = partialLoss - partialEnergy; maxCostY = possibleY

          }
          out.set(x, y, z, maxCostY)

        }
      }
    }

    return out
  }

  /**
   * thetaUnary is of size r x K, where is the number of regions
   * thetaPairwise is of size K x K
   */ //     X                Y                  Z
  def decodeFn(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], imageWidth: Int, imageHeight: Int, imageDepth: Int, debug: Boolean = false): NominalThreeDimMat[Int] = {

    val numRegions: Int = thetaUnary.rows
    val numClasses: Int = thetaUnary.cols

    assert(DISABLE_PAIRWISE || thetaPairwise.rows == numClasses)

    // Convert the image into a grid of Factorie variables
    val image: Buffer[Seq[Seq[Pixel]]] = new ArrayBuffer
    for (i <- 0 until imageWidth) { //TODO check if I am mixing up dimX and dimY 
      val row = new ArrayBuffer[ArrayBuffer[Pixel]]
      for (j <- 0 until imageHeight) { //TODO check if I am mixing up dimX and dimY 
        val stack = new ArrayBuffer[Pixel]
        for (k <- 0 until imageDepth) {
          stack += new Pixel(0) //0 is the initial value 
        }
        row += stack
      }

      image += row
    }

    class RegionVar(val score: Int) extends IntegerVariable(score)

    object PixelDomain extends DiscreteDomain(numClasses)

    class Pixel(i: Int) extends DiscreteVariable(i) { //i is just the initial value 
      def domain = PixelDomain
    }

    def getUnaryFactor(yi: Pixel, x: Int, y: Int, z: Int): Factor = {
      new Factor1(yi) { //TODO ask OT,  it appears that yi is not being used for anything. 
        val r = columnMajorIdx3d(x, y, z, imageWidth, imageHeight) //TODO check if I am mixing up dimX and dimY 
        def score(k: Pixel#Value) = thetaUnary(r, k.intValue)
      }
    }

    def getPairwiseFactor(yi: Pixel, yj: Pixel): Factor = { //Adapter method for the Factorie types Pixel to get the theta of their member values 
      new Factor2(yi, yj) {
        def score(i: Pixel#Value, j: Pixel#Value) = thetaPairwise(i.intValue, j.intValue)
      }
    }

    val indexedPixels = //I think this was just created in order to avoid for loops on the lines below 
      for {
        x <- 0 until image.size;
        y <- 0 until image(0).size;
        z <- 0 until image(0)(0).size
      } yield {
        ((x, y, z), image(x)(y)(z))
      }
    val regionIndexedPixels =
      indexedPixels.map {
        case ((x, y, z), pix) =>
          val r = columnMajorIdx3d(x, y, z, imageWidth, imageHeight)
          (r, pix)
      }

    val pixels: IndexedSeq[Pixel] = indexedPixels.map(_._2) //TODO understand this _._2 notation. I'm guessing this is just removing the (x,y) and just saving the other part in an array 

    val unaries: IndexedSeq[Factor] = indexedPixels.map {
      case ((x, y, z), pix) =>
        getUnaryFactor(pix, x, y, z)
    }

    val pairwise: IndexedSeq[Factor] =
      indexedPixels.flatMap {
        case ((x, y, z), pix) =>
          val factors = new ArrayBuffer[Factor]

          // (x, y) and (x, y+1)
          if (y < imageHeight - 1)
            factors ++= getPairwiseFactor(pix, image(x)(y + 1)(z)) //TODO Ask OT :  it looks like += would work here why use ++= 

          // (x, y) and (x+1, y)
          if (x < imageWidth - 1)
            factors ++= getPairwiseFactor(pix, image(x + 1)(y)(z))

          if (z < imageDepth - 1)
            factors ++= getPairwiseFactor(pix, image(x)(y)(z + 1))

          factors
      }

    val model = new ItemizedModel
    model ++= unaries
    if (!DISABLE_PAIRWISE) model ++= pairwise

    val maxIterations = if (DISABLE_PAIRWISE) 100 else 1000
    val maximizer = new MaximizeByMPLP(maxIterations)
    val assgn = maximizer.infer(pixels, model).mapAssignment //Where the magic happens 

    // Retrieve assigned labels from these pixels

    val imgMask: NominalThreeDimMat[Int] = new NominalThreeDimMat[Int](Vector(imageWidth, imageHeight, imageDepth), classes = (0 until numClasses).toList.toSet)

    for (x <- 0 until imageWidth) {
      for (y <- 0 until imageHeight) {
        for (z <- 0 until imageDepth) {
          imgMask.set(x, y, z, assgn(image(x)(y)(z)).intValue)
        }
      }
    }

    imgMask
  }

  /**
   * Takes as input:
   * - a weight vector of size: (f * K) + (K * K), i.e, unary features [xFeatureSize * numClasses] + pairwise features [numClasses * numClasses]
   * - feature size of x
   * - number of class labels
   * - padded - if padded, returns [I(K=0) w_0 ... I(K=0) w_k], else returns a matrix of size f x K: [w_1 w_2 ... w_k], i.e, each column is feature representation of some label
   */
  def unpackWeightVec(weightVec: Vector[Double], xFeatureSize: Int, numClasses: Int = 24, padded: Boolean = false): (DenseMatrix[Double], DenseMatrix[Double]) = {
    // Unary features
    val startIdx = 0
    val endIdx = xFeatureSize * numClasses
    val unaryFeatureVec = weightVec(startIdx until endIdx).toDenseVector // Stored as [|--f(k=0)--||--f(k=1)--| ... |--f(K=k)--|]
    val tempUnaryPot = unaryFeatureVec.toDenseMatrix.reshape(xFeatureSize, numClasses)

    val unaryPot =
      if (padded) {
        // Each column in this vector contains [I(K=0) w_0 ... I(K=0) w_k]
        val unaryPotPadded = DenseMatrix.zeros[Double](xFeatureSize * numClasses, numClasses)
        for (k <- 0 until numClasses) {
          val w = tempUnaryPot(::, k)
          val startIdx = k * xFeatureSize
          val endIdx = startIdx + xFeatureSize
          unaryPotPadded(startIdx until endIdx, k) := w
        }
        unaryPotPadded
      } else {
        tempUnaryPot
      }

    // Pairwise feature Vector
    val pairwisePot =
      if (DISABLE_PAIRWISE) {
        DenseMatrix.zeros[Double](0, 0)
      } else {
        val pairwiseFeatureVec = weightVec(endIdx until weightVec.size).toDenseVector
        assert(pairwiseFeatureVec.size == numClasses * numClasses)
        pairwiseFeatureVec.toDenseMatrix.reshape(numClasses, numClasses)
      }

    (unaryPot, pairwisePot)
  }

  def oracleFn(model: StructSVMModel[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]], xi: ThreeDimMat[Array[Double]], yi: NominalThreeDimMat[Int]): NominalThreeDimMat[Int] = {

    val numClasses = model.numClasses
    val numCols = xi.xDim
    val numRows = xi.yDim
    val numStack = xi.zDim
    val numROI = numRows * numCols * numStack
    val numDims = xi.get(0, 0, 0).length

    val weightVec = model.getWeights()

    // Unary is of size f x K, each column representing feature vector of class K
    // Pairwise if of size K * K
    val (unaryWeights, pairwiseWeights) = unpackWeightVec(weightVec, numDims, numClasses = numClasses, padded = false)
    assert(unaryWeights.rows == numDims)
    assert(unaryWeights.cols == numClasses)
    assert(DISABLE_PAIRWISE || pairwiseWeights.rows == pairwiseWeights.cols)
    assert(DISABLE_PAIRWISE || pairwiseWeights.rows == numClasses)

    val phi_Y: DenseMatrix[Double] = getUnaryFeatureMap_justX(xi) // Retrieves a f x r matrix representation of the original image, i.e, each column is the feature vector that region r
    val thetaUnary = phi_Y.t * unaryWeights // Returns a r x K matrix, where theta(r, k) is the unary potential of region r having label k
    //TODO check if this transpose makes sense 

    val thetaPairwise = pairwiseWeights

    // If yi is present, do loss-augmentation
    if (yi != null) {
      for (x <- 0 until yi.xDim) {
        for (y <- 0 until yi.yDim) {
          for (z <- 0 until yi.zDim) {
            val idx = columnMajorIdx3d(x, y, z, yi.xDim, yi.yDim)
            thetaUnary(idx, ::) := thetaUnary(idx, ::) + 1.0 / numROI //We are using a zero-one loss per y so here there are just constants
            // Loss augmentation step
            val k = yi.get(x, y, z)
            thetaUnary(idx, k) = thetaUnary(idx, k) - 1.0 / numROI //This is a zero loss b/c it cancels out the +1 for all non correct labels 
            //This zero one loss is repeated code from the lossFn. lossFn gets loss for  
            //     a whole image but inside it uses zero-one loss for pixel comparison 
            //     this should be a function so it is common between the two uses 
          }
        }
      }

    }

    /**
     * Parameter estimation
     */
    val startTime = System.currentTimeMillis()
    val decoded = decodeFn(thetaUnary, thetaPairwise, numCols, numRows, numStack, debug = false)
    val decodeTimeMillis = System.currentTimeMillis() - startTime
    
    //TODO add if debug == true for this test
    if ( yi != null) {
    print( if(decoded.isInverseOf(yi)) "[IsInv]" else "[NotInv]" +  "Decoding took : " + Math.round(decodeTimeMillis/1000) +"s")
    }
    
    return decoded
  }
  def predictFn(model: StructSVMModel[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]], xi: ThreeDimMat[Array[Double]]): NominalThreeDimMat[Int] = {
    return oracleFn(model, xi, yi = null)
  }

  //TODO ask someone if this is correct. 
  def predictFn_MaxPerPixel(model: StructSVMModel[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]], xi: ThreeDimMat[Array[Double]]): NominalThreeDimMat[Int] = {
    val numFeatures = xi.get(0, 0, 0).length

    assert(model.weights.size == (numFeatures * model.numClasses))

    var out: NominalThreeDimMat[Int] = new NominalThreeDimMat(Vector(xi.xDim, xi.yDim, xi.zDim), classes = ((0 to model.numClasses).toList toSet))

    for (x <- 0 until xi.xDim) {
      for (y <- 0 until xi.yDim) {
        for (z <- 0 until xi.zDim) {
          var leastCost = Double.MinValue
          var leastCostY = 0 //again assuming y_classes are consecutive integers
          for (possibleY <- 0 until model.numClasses) {
            val x_i = xi.get(x, y, z)

            val phi_i = DenseVector.zeros[Double](numFeatures * model.numClasses)
            val startIdx = numFeatures * possibleY //assumption of y_i being an int comes in here. Else this would have to have some static ordering 
            val endIdx = startIdx + numFeatures
            phi_i(startIdx until endIdx) := Vector(x_i)

            val partialEnergy = model.weights dot phi_i

            //TODO check the sign on this comparison 
            if (leastCost > (partialEnergy)) leastCost = partialEnergy; leastCostY = possibleY

          }
          out.set(x, y, z, leastCostY)

        }
      }
    }

    return out
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
    println(options)
    runStuff(options)
  }
  def runStuff(options: Map[String, String]) {

    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")
    val runLocally: Boolean = options.getOrElse("local", "true").toBoolean
    val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val msrcDir: String = "../data/generated"

    val appName: String = options.getOrElse("appname", "ImageSeg")

    val solverOptions: SolverOptions[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]] = new SolverOptions()
    solverOptions.roundLimit = options.getOrElse("roundLimit", "5").toInt // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean
    solverOptions.lambda = options.getOrElse("lambda", "0.01").toDouble
    solverOptions.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    solverOptions.doLineSearch = options.getOrElse("linesearch", "true").toBoolean
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean

    solverOptions.sample = options.getOrElse("sample", "frac")
    solverOptions.sampleFrac = options.getOrElse("samplefrac", "0.5").toDouble
    solverOptions.sampleWithReplacement = options.getOrElse("samplewithreplacement", "false").toBoolean

    solverOptions.enableManualPartitionSize = options.getOrElse("manualrddpart", "false").toBoolean
    solverOptions.NUM_PART = options.getOrElse("numpart", "2").toInt

    solverOptions.enableOracleCache = options.getOrElse("enableoracle", "false").toBoolean
    solverOptions.oracleCacheSize = options.getOrElse("oraclesize", "5").toInt

    solverOptions.debugInfoPath = options.getOrElse("debugpath", debugDir + "/imageseg-%d.csv".format(System.currentTimeMillis()))

    /**
     * Some local overrides
     */
    if (runLocally) {
      solverOptions.sampleFrac = 0.5
      solverOptions.enableOracleCache = false
      solverOptions.oracleCacheSize = 100
      solverOptions.stoppingCriterion = RoundLimitCriterion
      solverOptions.roundLimit = 5
      solverOptions.enableManualPartitionSize = true
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false

      solverOptions.debug = true
      solverOptions.debugMultiplier = 1
    }

    // (Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]], Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]]) 
    val (trainData, testData) = ThreeDimUtils.generateSomeData(20, 25, 3, 5, 2)
    solverOptions.numClasses = 2

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
        Some(sc.parallelize(testData.toSeq, solverOptions.NUM_PART))
      else
        Some(sc.parallelize(testData.toSeq))

    val trainDataRDD =
      if (solverOptions.enableManualPartitionSize)
        sc.parallelize(trainData, solverOptions.NUM_PART)
      else
        sc.parallelize(trainData)

    val trainer: StructSVMWithDBCFW[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]] =
      new StructSVMWithDBCFW[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]](
        trainDataRDD,
        this, //TODO Bookmark
        solverOptions)

    val model: StructSVMModel[ThreeDimMat[Array[Double]], NominalThreeDimMat[Int]] = trainer.trainModel()

    var avgTrainLoss = 0.0
   

    for (item <- trainData) {
      val prediction = model.predict(item.pattern)
      avgTrainLoss += lossFn(item.label, prediction)
    }
    avgTrainLoss = avgTrainLoss / trainData.size
    println("\nTRAINING: Avg Loss : " + avgTrainLoss + " numItems " + testData.size)
    //Test Error 
    avgTrainLoss = 0.0
     for (item <- testData) {
      val prediction = model.predict(item.pattern)
      avgTrainLoss += lossFn(item.label, prediction)
    }
    avgTrainLoss = avgTrainLoss / testData.size
    println("\nTest Avg Loss : " + avgTrainLoss + " numItems " + testData.size)

  }

}



    

  


