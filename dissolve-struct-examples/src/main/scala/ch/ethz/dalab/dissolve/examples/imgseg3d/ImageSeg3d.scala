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

object ImageSeg3d extends DissolveFunctions[ThreeDimMat[Array[Double]], ThreeDimMat[Int]] with Serializable {
  val DISABLE_PAIRWISE : Boolean = false
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

  // We assume classes are given by sequencial Int values 
  def getUnaryFeatureMap(yMat: ThreeDimMat[Int], xMat: ThreeDimMat[Array[Double]]): DenseMatrix[Double] = {
    assert(xMat.xDim == yMat.xDim)
    assert(xMat.yDim == yMat.yDim)
    assert(xMat.zDim == yMat.zDim)

    val numFeatures = xMat.get(0, 0, 0).length //TODO this hist feature size uniformity is not garateed inside my datastructure 
    val numClasses = yMat.classSet.size
    
    assert (yMat.classSet.toSeq.sum == (0 until numClasses).reduceLeft[Int](_+_) ) //Testing sequencial assumoption rufly 
    
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
  

  def getPairwiseFeatureMap(yMat: ThreeDimMat[Int], xMat: ThreeDimMat[Array[Double]]): DenseMatrix[Double] = {

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
  def featureFn(xMat: ThreeDimMat[Array[Double]], yMat: ThreeDimMat[Int]): Vector[Double] = {

   assert(xMat.xDim == yMat.xDim)
    assert(xMat.yDim == yMat.yDim)
    assert(xMat.zDim == yMat.zDim)

    val numDims = xMat.get(0, 0, 0).length //TODO this hist feature size uniformity is not garateed inside my datastructure 
    val numClasses = yMat.classSet.size

    val unaryFeatureSize = numDims * numClasses
    val pairwiseFeatureSize = numClasses * numClasses
    val phi = DenseVector.zeros[Double](unaryFeatureSize + pairwiseFeatureSize)//TODO if DISABLE_PAIRWISE == false then should the phi size not be reduced 

    // Unaries vector, of size f * K. Each column corresponds to a feature vector
    val unary = DenseVector.zeros[Double](numDims * numClasses)
    
     for (
      y <- 0 until xMat.xDim;
      x <- 0 until xMat.yDim;
      z <- 0 until xMat.zDim
    ) {  //TODO cant you jsut call  getUnaryFeatureMap()  for this ? 
        val label = yMat.get(x, y, z)
        val startIdx = label * numDims
        val endIdx = startIdx + numDims
        unary(startIdx until endIdx) := Vector(xMat.get(x,y,z)) + unary(startIdx until endIdx)
     }
    // Set Unary Features
    phi(0 until (numDims * numClasses)) := unary

    if (!DISABLE_PAIRWISE) { //TODO maybe this a global option 
      val pairwise = normalize(getPairwiseFeatureMap(yMat, xMat).toDenseVector) //TODO does this toDenseVector actually use proper columnIndex form ? 
      phi((numDims * numClasses) until phi.size) := pairwise
    }

    phi
  }
  
  
  
  /**
   * Loss function
   */
  def lossFn(yTruth: ThreeDimMat[Int], yPredict: ThreeDimMat[Int]): Double = { //TODO convert to new 3dmat class


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
        if (yTruth.get(x,y,z) == yPredict.get(x,y,z)) 0.0 else 1.0 / classFreqs.get(yTruth.get(x, y,z)).get  // Insert classFrequency back into the truthObject
      }

    loss.sum / (yTruth.xDim * yTruth.yDim * yTruth.zDim)
  }
  
  
  def oracleFn(model: StructSVMModel[ThreeDimMat[Array[Double]], ThreeDimMat[Int]], xi: ThreeDimMat[Array[Double]], yi: ThreeDimMat[Int]): ThreeDimMat[Int] = {
       
    //TODO yi is the truth for this particular xi, since in this algo we update a random xi constraint block at a time 
    
    val numFeatures = xi.get(0,0,0).length
    
    assert(model.weights.size == (numFeatures * yi.classSet.size))
     
    var out: ThreeDimMat[Int] = new ThreeDimMat(Vector(yi.xDim,yi.yDim,yi.zDim))
       
    
      for (x <- 0 until yi.xDim) {
      for (y <- 0 until yi.yDim) {
        for (z <- 0 until yi.zDim) {
           var maxCost = Double.MinValue
           var maxCostY = 0 //again assuming y_classes are consecutive integers
           for ( possibleY <- yi.classSet.toSeq) {
              val x_i = xi.get(x, y, z)
              
              val phi_i = DenseVector.zeros[Double](numFeatures * yi.classSet.size)
              val startIdx = numFeatures * possibleY //assumption of y_i being an int comes in here. Else this would have to have some static ordering 
              val endIdx = startIdx + numFeatures
              phi_i(startIdx until endIdx) := Vector(x_i)
              
              val partialEnergy = model.weights dot phi_i
            val partialLoss = (if (yi.get(x, y, z) == possibleY) 0.0 else 1.0 / yi.classFreq.get(yi.get(x, y, z)).get) / (yi.xDim * yi.yDim * yi.zDim) 
              if( maxCost <(  partialLoss - partialEnergy ))  maxCost =   partialLoss - partialEnergy ;  maxCostY = possibleY
   
           }
       out.set(x, y, z,maxCostY) 

        }
      }
    }
    
    
   
   return  out
  }
  
  //TODO ask someone if this is correct. 
    def predictFn(model: StructSVMModel[ThreeDimMat[Array[Double]], ThreeDimMat[Int]], xi: ThreeDimMat[Array[Double]]): ThreeDimMat[Int] = {
      val numFeatures = xi.get(0,0,0).length
    
    assert(model.weights.size == (numFeatures * xi.classSet.size))
     
    var out: ThreeDimMat[Int] = new ThreeDimMat(Vector(xi.xDim,xi.yDim,xi.zDim))
       
    
      for (x <- 0 until xi.xDim) {
      for (y <- 0 until xi.yDim) {
        for (z <- 0 until xi.zDim) {
           var leastCost = Double.MaxValue
           var leastCostY = 0 //again assuming y_classes are consecutive integers
           for ( possibleY <- 0 until model.numClasses) {
              val x_i = xi.get(x, y, z)
              
              val phi_i = DenseVector.zeros[Double](numFeatures * model.numClasses)
              val startIdx = numFeatures * possibleY //assumption of y_i being an int comes in here. Else this would have to have some static ordering 
              val endIdx = startIdx + numFeatures
              phi_i(startIdx until endIdx) := Vector(x_i)
              
              val partialEnergy = model.weights dot phi_i
         
              if( leastCost >( partialEnergy  ))  leastCost =  partialEnergy;  leastCostY = possibleY
   
           }
       out.set(x, y, z,leastCostY) 

        }
      }
    }
    
    
   
   return  out
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
    def runStuff (options : Map[String, String]){
    
    
    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")
    val runLocally: Boolean = options.getOrElse("local", "true").toBoolean
        val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val msrcDir: String = "../data/generated"

    val appName: String = options.getOrElse("appname", "ImageSeg")

    
    
    
    
    
     val solverOptions: SolverOptions[ThreeDimMat[Array[Double]], ThreeDimMat[Int]] = new SolverOptions()
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

      solverOptions.debug = false
      solverOptions.debugMultiplier = 1
    }

    
    // (Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]], Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]]) 
    val (trainData, testData) = ThreeDimUtils.generateSomeData(15, 25, 3, 5, 0.1)

    
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

    val trainer: StructSVMWithDBCFW[ThreeDimMat[Array[Double]], ThreeDimMat[Int]] =
      new StructSVMWithDBCFW[ThreeDimMat[Array[Double]], ThreeDimMat[Int]](
        trainDataRDD,
        this, //TODO Bookmark
        solverOptions)

    val model: StructSVMModel[ThreeDimMat[Array[Double]], ThreeDimMat[Int]] = trainer.trainModel()

  }

}

    

  


