package ch.ethz.dal.dbcfw.optimization

import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import org.apache.spark.SparkContext
import ch.ethz.dal.dbcfw.classification.Types._
import org.apache.spark.SparkContext._
import org.apache.log4j.Logger
import scala.collection.mutable

/**
 * LogHelper is a trait you can mix in to provide easy log4j logging
 * for your scala classes.
 */
/*trait LogHelper {
  val loggerName = this.getClass.getName
  lazy val logger = Logger.getLogger(loggerName)
}*/

class DBCFWSolver(
  @transient val sc: SparkContext,
  val data: Vector[LabeledObject],
  val featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVect, 
  val lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossVal, 
  val oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Lab, 
  val predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
  val solverOptions: SolverOptions,
  val miniBatchEnabled: Boolean) extends Serializable {

  /**
   * This runs on the Master node, and each round triggers a map-reduce job on the workers
   */
  def optimize(): (StructSVMModel, String) = {

    // autoconfigure parameters
    val NUM_DECODING_SAMPLES = 5
    val NUM_COMMN_SAMPLES = 5 // Time taken for a single round of communication

    val debugSb: StringBuilder = new StringBuilder()

    val d: Int = featureFn(data(0).label, data(0).pattern).size
    // Let the initial model contain zeros for all weights
    var globalModel: StructSVMModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    /**
     *  Create two RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     */
    val indexedTrainData: Array[(Index, LabeledObject)] = (0 until data.size).toArray.zip(data.toArray)
    val indexedPrimals: Array[(Index, PrimalInfo)] = (0 until data.size).toArray.zip(
      Array.fill(data.size)((DenseVector.zeros[Double](d), 0.0)) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
      )

    val indexedTrainDataRDD: RDD[(Index, LabeledObject)] = sc.parallelize(indexedTrainData, solverOptions.NUM_PART)
    var indexedPrimalsRDD: RDD[(Index, PrimalInfo)] = sc.parallelize(indexedPrimals, solverOptions.NUM_PART)

    indexedPrimalsRDD.checkpoint()

    /**
     * Fix parameters to perform sampling.
     * Use can either specify:
     * a) "count" - Eqv. to 'H' in paper. Number of points to sample in each round.
     * or b) "perc" - Fraction of dataset to sample \in [0.0, 1.0]
     */
    val sampleFrac: Double = {
      if (solverOptions.sample == "frac")
        solverOptions.sampleFrac
      else if (solverOptions.sample == "count")
        math.min(solverOptions.H / data.size, 1.0)
      else {
        println("[WARNING] %s is not a valid option. Reverting to sampling 50% of the dataset")
        0.5
      }
    }

    println("Beginning training of %d data points in %d passes with lambda=%f".format(data.size, solverOptions.numPasses, solverOptions.lambda))

    var avgDecodeTime: Double = 0.0
    var avgCommunicationTime: Double = 0.0
    /**
     * Monitoring round
     * In this mode, the optimal H and numPasses is calculated based on:
     * a. Time taken for decoding
     * b. Time taken for a single round of communication
     */
    if (solverOptions.autoconfigure) {

      /**
       * Obtain average time required to decode
       */
      var decodeTimings =
        for (i <- 0 until NUM_DECODING_SAMPLES) yield {
          var randModel: StructSVMModel = new StructSVMModel(DenseVector.rand(d), Math.random(), DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)
          var sampled_i = util.Random.nextInt(data.size)

          val startDecodingTime = System.currentTimeMillis()
          oracleFn(randModel, data(sampled_i).label, data(sampled_i).pattern)
          val endDecodingTime = System.currentTimeMillis()

          endDecodingTime - startDecodingTime
        }
      avgDecodeTime = decodeTimings.reduce((t1, t2) => t1 + t2).toDouble / NUM_DECODING_SAMPLES

      /**
       * Obtain average time required to finish 1 round of communication
       *
       * Run a dummy map-reduce job to get the timings
       */
      def dummyOracle(model: StructSVMModel, yi: Vector[Double], xi: Matrix[Double]): Vector[Double] = DenseVector.ones[Double](yi.size)

      val communicationTimings =
        for (i <- 0 until NUM_COMMN_SAMPLES) yield {

          // Run Mapper
          val temp: RDD[(StructSVMModel, Array[(Index, PrimalInfo)], StructSVMModel)] = indexedTrainDataRDD.sample(solverOptions.sampleWithReplacement, sampleFrac, solverOptions.randSeed)
            .join(indexedPrimalsRDD)
            .mapPartitions(x => mapper(x, globalModel, featureFn, lossFn, dummyOracle,
              predictFn, solverOptions, miniBatchEnabled), preservesPartitioning = true)

          // Finish Reducer. Thus, finish one round of communication
          val startCommunicationTime = System.currentTimeMillis()
          val reducedData: (StructSVMModel, RDD[(Index, PrimalInfo)]) = reducer(temp, indexedPrimalsRDD, globalModel, d, beta = 1.0)
          val endCommunicationTime = System.currentTimeMillis()

          endCommunicationTime - startCommunicationTime
        }
      avgCommunicationTime = communicationTimings.reduce((t1, t2) => t1 + t2).toDouble / NUM_COMMN_SAMPLES

      /**
       * Given the average time to decode samples and communicate, figure out values of H and numPasses
       */
      /*println("Average decoding time: %f".format(avgDecodeTime))
      println("Average communication time: %f".format(avgCommunicationTime))*/

    }

    // logger.info("[DATA] round,time,train_error,test_error")
    val startTime = System.currentTimeMillis()
    debugSb ++= "round,time,primal,dual,gap,train_error,test_error\n"

    for (roundNum <- 1 to solverOptions.numPasses) {

      val temp: RDD[(StructSVMModel, Array[(Index, PrimalInfo)], StructSVMModel)] = indexedTrainDataRDD.sample(solverOptions.sampleWithReplacement, sampleFrac, solverOptions.randSeed)
        .join(indexedPrimalsRDD)
        .mapPartitions(x => mapper(x, globalModel, featureFn, lossFn, oracleFn,
          predictFn, solverOptions, miniBatchEnabled), preservesPartitioning = true)

      val reducedData: (StructSVMModel, RDD[(Index, PrimalInfo)]) = reducer(temp, indexedPrimalsRDD, globalModel, d, beta = 1.0)

      // Update the global model and the primal for each i
      globalModel = reducedData._1
      indexedPrimalsRDD = reducedData._2

      val elapsedTime = (System.currentTimeMillis() - startTime).toDouble / 1000.0

      val trainError = SolverUtils.averageLoss(data, lossFn, predictFn, globalModel)
      val testError = SolverUtils.averageLoss(solverOptions.testData, lossFn, predictFn, globalModel)

      // Obtain duality gap after each communication round
      val debugModel: StructSVMModel = globalModel.clone()
      val f = -SolverUtils.objectiveFunction(debugModel.getWeights, debugModel.getEll, solverOptions.lambda)
      val gapTup = SolverUtils.dualityGap(data, featureFn, lossFn, oracleFn, debugModel, solverOptions.lambda)
      val gap = gapTup._1
      val primal = f + gap

      // logger.info("[DATA] %d,%f,%f,%f\n".format(roundNum, elapsedTime, trainError, testError))
      println("[Round #%d] Train loss = %f, Test loss = %f, Primal = %f, Gap = %f\n".format(roundNum, trainError, testError, primal, gap))
      val curTime = (System.currentTimeMillis() - startTime) / 1000
      debugSb ++= "%d,%d,%f,%f,%f,%f,%f\n".format(roundNum, curTime, primal, f, gap, trainError, testError)
    }

    println("Average decoding time: %f".format(avgDecodeTime))
    println("Average communication time: %f".format(avgCommunicationTime))

    (globalModel, debugSb.toString())
  }

  /**
   * Takes as input a set of data and builds a SSVM model trained using BCFW
   */
  def mapper(dataIterator: Iterator[(Index, (LabeledObject, PrimalInfo))],
    localModel: StructSVMModel,
    featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVect, 
    lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossVal, 
    oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Lab, 
    predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
    solverOptions: SolverOptions,
    miniBatchEnabled: Boolean): Iterator[(StructSVMModel, Array[(Index, PrimalInfo)], StructSVMModel)] = {

    val prevModel: StructSVMModel = localModel.clone()

    val numPasses = solverOptions.numPasses
    val lambda = solverOptions.lambda
    val debugOn: Boolean = solverOptions.debug
    val xldebug: Boolean = solverOptions.xldebug

    /**
     * Reorganize data for training
     */
    val zippedData: Array[(Index, (LabeledObject, PrimalInfo))] = dataIterator.toArray.sortBy(_._1)
    val data: Array[LabeledObject] = zippedData.map(x => x._2._1)
    val globalDataIdx: Array[Index] = zippedData.map(x => x._1)
    // Mapping of indexMapping(localIndex) -> globalIndex
    val localToGlobal: Array[Index] = zippedData.map(x => x._1)

    // Recursively convert an array to an immutable Map, mapping array elements("global index") with their indices("local index")
    /*def convertArrayToMap(arr: Array[Index], pos: Index): Map[Index, Index] =
      if (arr.length > 0)
        convertArrayToMap(arr.drop(1), pos + 1) + (arr(0) -> pos)
      else
        Map.empty[Index, Index]*/

    // Alternate implementation - Use mutable maps. Immutable causes stack overflow
    val globalToLocal: mutable.Map[Index, Index] = mutable.Map.empty[Index, Index]
    for (ele <- localToGlobal.zipWithIndex)
      globalToLocal(ele._1) = ele._2

    val maxOracle = oracleFn
    val phi = featureFn
    // Number of dimensions of \phi(x, y)
    val d: Int = localModel.getWeights().size

    // Only to keep track of the \Delta localModel
    val deltaLocalModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    val eps: Double = 2.2204E-16

    var k: Int = 0
    val n: Int = data.size

    val wMat: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, n)
    val ellMat: DenseVector[Double] = DenseVector.zeros[Double](n)

    // Copy w_i's and l_i's into local wMat and ellMat
    for (i <- 0 until n) {
      wMat(::, i) := zippedData(i)._2._2._1
      ellMat(i) = zippedData(i)._2._2._2
    }
    val prevWMat: DenseMatrix[Double] = wMat.copy
    val prevEllMat: DenseVector[Double] = ellMat.copy

    var ell: Double = localModel.getEll()
    localModel.updateEll(0.0)

    // Initialization in case of Weighted Averaging
    var wAvg: DenseVector[Double] =
      if (solverOptions.doWeightedAveraging)
        DenseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    for ((datapoint, globalIdx) <- data.zip(globalDataIdx)) {

      // Convert globalIdx to localIdx a.k.a "i"
      val i: Index = globalToLocal(globalIdx)

      // 1) Pick example
      val pattern: Matrix[Double] = datapoint.pattern
      val label: Vector[Double] = datapoint.label

      // 2) Solve loss-augmented inference for point i
      val ystar_i: Vector[Double] =
        if (!miniBatchEnabled)
          maxOracle(localModel, label, pattern)
        else
          maxOracle(prevModel, label, pattern)

      // 3) Define the update quantities
      val psi_i: Vector[Double] = phi(label, pattern) - phi(ystar_i, pattern)
      val w_s: Vector[Double] = psi_i :* (1.0 / (n * lambda))
      val loss_i: Double = lossFn(label, ystar_i)
      val ell_s: Double = (1.0 / n) * loss_i

      // 4) Get step-size gamma
      val gamma: Double =
        if (solverOptions.doLineSearch) {
          val thisModel = if (miniBatchEnabled) prevModel else localModel
          val gamma_opt = (thisModel.getWeights().t * (wMat(::, i) - w_s) - ((ellMat(i) - ell_s) * (1.0 / lambda))) /
            ((wMat(::, i) - w_s).t * (wMat(::, i) - w_s) + eps)
          max(0.0, min(1.0, gamma_opt))
        } else {
          (2.0 * n) / (k + 2.0 * n)
        }

      // 5, 6, 7, 8) Update the weights of the model
      if (miniBatchEnabled) {
        wMat(::, i) := wMat(::, i) * (1.0 - gamma) + (w_s * gamma)
        ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
        deltaLocalModel.updateWeights(localModel.getWeights() + (wMat(::, i) - prevWMat(::, i)))
        deltaLocalModel.updateEll(localModel.getEll() + (ellMat(i) - prevEllMat(i)))
      } else {
        // In case of CoCoA
        val tempWeights1: Vector[Double] = localModel.getWeights() - wMat(::, i)
        localModel.updateWeights(tempWeights1)
        deltaLocalModel.updateWeights(tempWeights1)
        wMat(::, i) := (wMat(::, i) * (1.0 - gamma)) + (w_s * gamma)
        val tempWeights2: Vector[Double] = localModel.getWeights() + wMat(::, i)
        localModel.updateWeights(tempWeights2)
        deltaLocalModel.updateWeights(tempWeights2)

        ell = ell - ellMat(i)
        ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
        ell = ell + ellMat(i)
      }

      // 9) Optionally update the weighted average
      if (solverOptions.doWeightedAveraging) {
        val rho: Double = 2.0 / (k + 2.0)
        wAvg = (wAvg * (1.0 - rho)) + (localModel.getWeights * rho)
        lAvg = (lAvg * (1.0 - rho)) + (ell * rho)
      }

      k = k + 1

    }

    if (solverOptions.doWeightedAveraging) {
      localModel.updateWeights(wAvg)
      localModel.updateEll(lAvg)
    } else {
      localModel.updateEll(ell)
    }

    val localIndexedDeltaPrimals: Array[(Index, PrimalInfo)] = zippedData.map(_._1).map(k => (k, (wMat(::, globalToLocal(k)) - prevWMat(::, globalToLocal(k)),
      ellMat(globalToLocal(k)) - prevEllMat(globalToLocal(k)))))

    // If this flag is set, return only the change in w's
    localModel.updateWeights(localModel.getWeights() - prevModel.getWeights())
    localModel.updateEll(localModel.getEll() - prevModel.getEll())

    // Finally return a single element iterator
    { List.empty[(StructSVMModel, Array[(Index, PrimalInfo)], StructSVMModel)] :+ (localModel, localIndexedDeltaPrimals, deltaLocalModel) }.iterator
  }

  /**
   * Takes as input a number of SVM Models, along with Primal information for each data point, and combines them into a single Model and Primal block
   */
  def reducer( // sc: SparkContext,
    zippedModels: RDD[(StructSVMModel, Array[(Index, PrimalInfo)], StructSVMModel)], // The optimize step returns k blocks. Each block contains (\Delta LocalModel, [\Delta PrimalInfo_i]).
    oldPrimalInfo: RDD[(Index, PrimalInfo)],
    oldGlobalModel: StructSVMModel,
    d: Int,
    beta: Double): (StructSVMModel, RDD[(Index, PrimalInfo)]) = {

    val k: Double = zippedModels.count.toDouble // This refers to the number of localModels generated

    // Here, map is applied k(=#workers) times
    val sumDeltaWeights =
      zippedModels.map(model => model._1.getWeights()).reduce((deltaWeightA, deltaWeightB) => deltaWeightA + deltaWeightB)
    val sumDeltaElls =
      zippedModels.map(model => model._1.getEll).reduce((ellA, ellB) => ellA + ellB)

    /**
     * Create the new global model
     */
    val newGlobalModel = new StructSVMModel(oldGlobalModel.getWeights() + (sumDeltaWeights / k) * beta,
      oldGlobalModel.getEll() + (sumDeltaElls / k) * beta,
      DenseVector.zeros(d),
      oldGlobalModel.featureFn,
      oldGlobalModel.lossFn,
      oldGlobalModel.oracleFn,
      oldGlobalModel.predictFn)

    /**
     * Merge all the w_i's and l_i's
     *
     * First flatMap returns a [newDeltaPrimalInfo_k]. This is applied k times, returns a sequence of n deltaPrimalInfos
     *
     * By doing a right outer join, we ensure that all the indices are retained, even in case data points are sampled
     *
     * After join, we have a sequence of (Index, (PrimalInfo_A, PrimalInfo_B))
     * where PrimalInfo_A = PrimalInfo_i at t-1
     * and   PrimalInfo_B = \Delta PrimalInfo_i
     */
    val indexedPrimals: RDD[(Index, PrimalInfo)] =
      zippedModels
        .flatMap { case (model, primals, debugModel) => primals }
        .rightOuterJoin(oldPrimalInfo)
        .map {
          case (idx, (Some((newW, newEll)), (prevW, prevEll))) =>
            (idx, (prevW + (newW * (beta / k)),
              prevEll + (newEll * (beta / k))))
          case (idx, (None, (prevW, prevEll))) => (idx, (prevW, prevEll))
        }

    // indexedPrimals isn't materialized till an RDD action is called. Force this by calling one.
    indexedPrimals.checkpoint()
    println(indexedPrimals.isCheckpointed)
    indexedPrimals.first()

    (newGlobalModel, indexedPrimals.sortByKey(true, 1))
  }

}