package ch.ethz.dalab.dissolve.optimization

import scala.collection.mutable.MutableList
import scala.reflect.ClassTag

import org.apache.spark.HashPartitioner
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.rdd.RDD

import breeze.linalg.DenseVector
import breeze.linalg.SparseVector
import breeze.linalg.Vector
import breeze.linalg.max
import breeze.linalg.min
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.Types.BoundedCacheList
import ch.ethz.dalab.dissolve.classification.Types.Index
import ch.ethz.dalab.dissolve.classification.Types.PrimalInfo
import ch.ethz.dalab.dissolve.regression.LabeledObject

/**
 * Train a structured SVM using the actual distributed dissolve^struct solver.
 * This uses primal dual Block-Coordinate Frank-Wolfe solver (BCFW), distributed
 * via the CoCoA framework (Communication-Efficient Distributed Dual Coordinate Ascent)
 *
 * @param <X> type for the data examples
 * @param <Y> type for the labels of each example
 */
class DBCFWSolverTuned[X, Y](
  val data: RDD[LabeledObject[X, Y]],
  val dissolveFunctions: DissolveFunctions[X, Y],
  val solverOptions: SolverOptions[X, Y],
  val miniBatchEnabled: Boolean = false) extends Serializable {

  /**
   * 
   * Some case classes to make code more readable
   */

  case class HelperFunctions[X, Y](featureFn: (X, Y) => Vector[Double],
                                   lossFn: (Y, Y) => Double,
                                   oracleFn: (StructSVMModel[X, Y], X, Y) => Y,
                                   predictFn: (StructSVMModel[X, Y], X) => Y)

  // Input to the mapper: idx -> DataShard
  case class InputDataShard[X, Y](labeledObject: LabeledObject[X, Y],
                                  primalInfo: PrimalInfo,
                                  cache: Option[BoundedCacheList[Y]])

  // Output of the mapper: idx -> ProcessedDataShard
  case class ProcessedDataShard[X, Y](primalInfo: PrimalInfo,
                                      cache: Option[BoundedCacheList[Y]],
                                      localSummary: Option[LocalSummary[X, Y]])

  case class LocalSummary[X, Y](deltaLocalModel: StructSVMModel[X, Y],
                                deltaLocalK: Vector[Int])

  // Experimental data
  case class RoundEvaluation(roundNum: Int,
                             elapsedTime: Double,
                             primal: Double,
                             dual: Double,
                             dualityGap: Double,
                             trainError: Double,
                             testError: Double) {
    override def toString(): String = "%d,%f,%f,%f,%f,%f,%f"
      .format(roundNum, elapsedTime, primal, dual, dualityGap, trainError, testError)
  }

  /**
   * This runs on the Master node, and each round triggers a map-reduce job on the workers
   */
  def optimize()(implicit m: ClassTag[Y]): (StructSVMModel[X, Y], String) = {

    val startTime = System.currentTimeMillis()

    val sc = data.context

    val debugSb: StringBuilder = new StringBuilder()

    val samplePoint = data.first()
    val dataSize = data.count().toInt
    val testDataSize = if (solverOptions.testDataRDD.isDefined) solverOptions.testDataRDD.get.count().toInt else 0

    val verboseDebug: Boolean = false

    val d: Int = dissolveFunctions.featureFn(samplePoint.pattern, samplePoint.label).size
    // Let the initial model contain zeros for all weights
    // Global model uses Dense Vectors by default
    
    if(solverOptions.initWithEmpiricalTransProb){
      
      val tmp =DenseVector(solverOptions.initWeight)
      print("##Using Init Weight: "+tmp)
    }
    
    
    var globalModel: StructSVMModel[X, Y] = new StructSVMModel[X, Y]( if(solverOptions.initWithEmpiricalTransProb) DenseVector(solverOptions.initWeight) else DenseVector.zeros(d), 0.0,
      DenseVector.zeros(d), dissolveFunctions, solverOptions.numClasses)

    val numPartitions: Int =
      data.partitions.size

    val beta: Double = 1.0

    val helperFunctions: HelperFunctions[X, Y] = HelperFunctions(dissolveFunctions.featureFn,
      dissolveFunctions.lossFn,
      dissolveFunctions.oracleFn,
      dissolveFunctions.predictFn)

    /**
     *  Create four RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     *  3. indexedCacheRDD (Index, BoundedCacheList)
     *  4. indexedLocalProcessedData (Index, LocallyProcessedData)
     *  all of which are partitioned similarly
     */

    /*
     * zipWithIndex calls getPartitions. But, partitioning happens in the future.
     * This causes a race condition.
     * See bug: https://issues.apache.org/jira/browse/SPARK-4433
     * 
     * Making do with work-around
     * 
    val indexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      data.zipWithIndex()
        .map {
          case (labeledObject, idx) =>
            (idx.toInt, labeledObject)
        }
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()
    * 
    */

    // The work-around for bug SPARK-4433
    val zippedIndexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      data.zipWithIndex()
        .map {
          case (labeledObject, idx) =>
            (idx.toInt, labeledObject)
        }
    zippedIndexedTrainDataRDD.count()

    val indexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      zippedIndexedTrainDataRDD
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()

    val indexedPrimals: Array[(Index, PrimalInfo)] = (0 until dataSize).toArray.zip(
      // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
      Array.fill(dataSize)((
        if (solverOptions.sparse) // w_i can be either Sparse or Dense 
          SparseVector.zeros[Double](d)
        else
          DenseVector.zeros[Double](d),
        0.0)))
    var indexedPrimalsRDD: RDD[(Index, PrimalInfo)] =
      sc.parallelize(indexedPrimals)
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()

    // For each Primal (i.e, Index), cache a list of Decodings (i.e, Y's)
    // If cache is disabled, add an empty array. This immediately drops the joins later on and saves time in communicating an unnecessary RDD.
    val indexedCache: Array[(Index, BoundedCacheList[Y])] =
      if (solverOptions.enableOracleCache)
        (0 until dataSize).toArray.zip(
          Array.fill(dataSize)(MutableList[Y]()) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
          )
      else
        Array[(Index, BoundedCacheList[Y])]()
    var indexedCacheRDD: RDD[(Index, BoundedCacheList[Y])] =
      sc.parallelize(indexedCache)
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()

    var indexedLocalProcessedData: RDD[(Index, ProcessedDataShard[X, Y])] = null

    val kAccum = DenseVector.zeros[Int](numPartitions)

    debugSb ++= "# indexedTrainDataRDD.partitions.size=%d\n".format(indexedTrainDataRDD.partitions.size)
    debugSb ++= "# indexedPrimalsRDD.partitions.size=%d\n".format(indexedPrimalsRDD.partitions.size)
    debugSb ++= "# sc.getExecutorStorageStatus.size=%d\n".format(sc.getExecutorStorageStatus.size)

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
        math.min(solverOptions.H / dataSize, 1.0)
      else {
        println("[WARNING] %s is not a valid option. Reverting to sampleFrac = 0.5".format(solverOptions.sample))
        0.5
      }
    }

    /**
     * In case of weighted averaging, start off with an all-zero (wAvg, lAvg)
     */
    var wAvg: Vector[Double] =
      if (solverOptions.doWeightedAveraging)
        DenseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    var weightedAveragesOfPrimals: PrimalInfo =
      if (solverOptions.doWeightedAveraging)
        (DenseVector.zeros(d), 0.0)
      else null

    var iterCount: Int = 0

    def getLatestModel(): StructSVMModel[X, Y] = {
      val debugModel: StructSVMModel[X, Y] = globalModel.clone()
      if (solverOptions.doWeightedAveraging) {
        debugModel.updateWeights(weightedAveragesOfPrimals._1)
        debugModel.updateEll(weightedAveragesOfPrimals._2)
      }
      debugModel
    }

    def getLatestGap(): Double = {
      val debugModel: StructSVMModel[X, Y] = getLatestModel()
      val gap = SolverUtils.dualityGap(data, dissolveFunctions, debugModel, solverOptions.lambda, dataSize)
      gap._1
    }

    def evaluateModel(model: StructSVMModel[X, Y], roundNum: Int = 0): RoundEvaluation = {
      val dual = -SolverUtils.objectiveFunction(model.getWeights(), model.getEll(), solverOptions.lambda)
      val dualityGap = SolverUtils.dualityGap(data, dissolveFunctions, model, solverOptions.lambda, dataSize)._1
      val primal = dual + dualityGap

      val trainError = SolverUtils.averageLoss(data, dissolveFunctions, model, dataSize)
      val testError =
        if (solverOptions.testDataRDD.isDefined)
          SolverUtils.averageLoss(solverOptions.testDataRDD.get, dissolveFunctions, model, testDataSize)
        else
          0.00

      val elapsedTime = getElapsedTimeSecs()

      println("[%.3f] Round = %d, Gap = %f, Primal = %f, Dual = %f, TrainLoss = %f, TestLoss = %f"
        .format(elapsedTime, roundNum, dualityGap, primal, dual, trainError, testError))
      
        if(dualityGap<0){
          println("# Neg Gap after "+gammaZeroInARow+"gamma < 0");
        }
          //assert(dualityGap>0)
      def bToS( a:Boolean)={if(a)"t"else"f"}
        
        println("#RoundProgTag# ,%d, %s , %s , %.3f, %d, %f, %f, %f, %f, %f , %.2f, %s, %s, %s, %d, %s, %s, %s, %d, %s, %s, %.3f, %d, %d, %s, %s, %s, %.3f, %d, %d, %.3f,%s,%.3f"
        .format(solverOptions.startTime, solverOptions.runName,solverOptions.gitVersion,elapsedTime, roundNum, dualityGap, primal,
            dual, trainError, testError,solverOptions.sampleFrac, if(solverOptions.doWeightedAveraging) "t" else "f", 
            if(solverOptions.onlyUnary) "t" else "f" ,if(solverOptions.squareSLICoption) "t" else "f" , solverOptions.superPixelSize, solverOptions.dataSetName, if(solverOptions.trainTestEqual)"t" else "f",
            solverOptions.inferenceMethod,solverOptions.dbcfwSeed, if(solverOptions.dataGenGreyOnly) "t" else "f", if(solverOptions.compPerPixLoss) "t" else "f", solverOptions.dataGenNeighProb, solverOptions.featHistSize,
            solverOptions.featCoOcurNumBins, if(solverOptions.useLoopyBP) "t" else "f", if(solverOptions.useMPLP) "t" else "f", bToS(solverOptions.slicNormalizePerClust), solverOptions.dataGenOsilNoise, solverOptions.dataRandSeed,
            solverOptions.dataGenHowMany,solverOptions.slicCompactness,bToS(solverOptions.putLabelIntoFeat),solverOptions.dataAddedNoise
            ))
        //TODO need to add expID tag, maybe git Version 

   
        
    
        
      RoundEvaluation(roundNum, elapsedTime, primal, dual, dualityGap, trainError, testError)
    }

    println("Beginning training of %d data points in %d passes with lambda=%f".format(dataSize, solverOptions.roundLimit, solverOptions.lambda))

    debugSb ++= "round,time,primal,dual,gap,train_error,test_error\n"

    def getElapsedTimeSecs(): Double = ((System.currentTimeMillis() - startTime) / 1000.0)

    /**
     * ==== Begin Training rounds ====
     */
    Stream.from(1)
      .takeWhile {
        roundNum =>
          val continueExecution =
            solverOptions.stoppingCriterion match {
              case RoundLimitCriterion => roundNum <= solverOptions.roundLimit
              case TimeLimitCriterion  => getElapsedTimeSecs() < solverOptions.timeLimit
              case GapThresholdCriterion =>
                // Calculating duality gap is really expensive. So, check ever gapCheck rounds
                if (roundNum % solverOptions.gapCheck == 0)
                  getLatestGap() > solverOptions.gapThreshold
                else
                  true
              case _ => throw new Exception("Unrecognized Stopping Criterion")
            }

          if (solverOptions.debug && (!(continueExecution || (roundNum - 1 % solverOptions.debugMultiplier == 0)) || roundNum == 1)) {
            // Force evaluation of model in 2 cases - Before beginning the very first round, and after the last round
            debugSb ++= evaluateModel(getLatestModel(), if (roundNum == 1) 0 else roundNum) + "\n"
          }

          continueExecution
      }
      .foreach {
        roundNum =>

          /**
           * Step 1 - Create a joint RDD containing all information of idx -> (data, primals, cache)
           */
          
          val indexedJointData: RDD[(Index, InputDataShard[X, Y])] = if(solverOptions.dbcfwSeed==(-1) )
            indexedTrainDataRDD
              .sample(solverOptions.sampleWithReplacement, sampleFrac)   
              .join(indexedPrimalsRDD)
              .leftOuterJoin(indexedCacheRDD)
              .mapValues { // Because mapValues preserves partitioning
                case ((labeledObject, primalInfo), cache) =>
                  InputDataShard(labeledObject, primalInfo, cache)
              }
          else
indexedTrainDataRDD
              .sample(solverOptions.sampleWithReplacement, sampleFrac,solverOptions.dbcfwSeed)  
              .join(indexedPrimalsRDD)
              .leftOuterJoin(indexedCacheRDD)
              .mapValues { // Because mapValues preserves partitioning
                case ((labeledObject, primalInfo), cache) =>
                  InputDataShard(labeledObject, primalInfo, cache)
              }
          
          /*println("indexedTrainDataRDD = " + indexedTrainDataRDD.count())
          println("indexedJointData.count = " + indexedJointData.count())
          println("indexedPrimalsRDD.count = " + indexedPrimalsRDD.count())
          println("indexedCacheRDD.count = " + indexedCacheRDD.count())*/

          /**
           * Step 2 - Map each partition to produce: idx -> (newPrimals, newCache, optionalModel)
           * Note that the optionalModel column is sparse. There exist only `numPartitions` of them in the RDD.
           */

          // if (indexedLocalProcessedData != null)
          // indexedLocalProcessedData.unpersist(false)

          indexedLocalProcessedData =
            indexedJointData.mapPartitionsWithIndex(
              (idx, dataIterator) =>
                mapper((idx, numPartitions),
                  dataIterator,
                  helperFunctions,
                  solverOptions,
                  globalModel,
                  dataSize,
                  kAccum),
              preservesPartitioning = true)
              .cache()

          /**
           * Step 2.5 - A long lineage may cause a StackOverFlow error in the JVM.
           * So, trigger a checkpointing once in a while.
           */
          if (roundNum % solverOptions.checkpointFreq == 0) {
            indexedPrimalsRDD.checkpoint()
            indexedCacheRDD.checkpoint()
            indexedLocalProcessedData.checkpoint()
          }

          /**
           * Step 3a - Obtain the new global model
           * Collect models from all partitions and compute the new model locally on master
           */

          val localSummaryList =
            indexedLocalProcessedData
              .flatMapValues(_.localSummary)
              .values
              .collect()

          //TODO remove, this print line is here to investigate the .reduceLeft empty error 
          // println("#d localSummaryList.size=%d".format(localSummaryList.size))
          val sumDeltaWeightsAndEll =
            localSummaryList
              .map {
                case summary =>
                  val model = summary.deltaLocalModel
                  (model.getWeights(), model.getEll())
              }.reduce(
                (model1, model2) =>
                  (model1._1 + model2._1, model1._2 + model2._2))

          val deltaK: Vector[Int] = localSummaryList
            .map(_.deltaLocalK)
            .reduce((x, y) => x + y)
          kAccum += deltaK

          val newGlobalModel = globalModel.clone()
          newGlobalModel.updateWeights(globalModel.getWeights() + sumDeltaWeightsAndEll._1 * (beta / numPartitions))
          newGlobalModel.updateEll(globalModel.getEll() + sumDeltaWeightsAndEll._2 * (beta / numPartitions))
          globalModel = newGlobalModel

          /**
           * Step 3b - Obtain the new set of primals
           */

          val newPrimalsRDD = indexedLocalProcessedData
            .mapValues(_.primalInfo)

          indexedPrimalsRDD = indexedPrimalsRDD
            .leftOuterJoin(newPrimalsRDD)
            .mapValues {
              case ((prevW, prevEll), Some((newW, newEll))) =>
                (prevW + (newW * (beta / numPartitions)),
                  prevEll + (newEll * (beta / numPartitions)))
              case ((prevW, prevEll), None) => (prevW, prevEll)
            }.cache()

          /**
           * Step 3c - Obtain the new cache values
           */

          val newCacheRDD = indexedLocalProcessedData
            .mapValues(_.cache)

          indexedCacheRDD = indexedCacheRDD
            .leftOuterJoin(newCacheRDD)
            .mapValues {
              case (oldCache, Some(newCache)) => newCache.get
              case (oldCache, None)           => oldCache
            }.cache()

          /**
           * Debug info
           */
          // Obtain duality gap after each communication round
          val debugModel: StructSVMModel[X, Y] = globalModel.clone()
          if (solverOptions.doWeightedAveraging) {
            debugModel.updateWeights(weightedAveragesOfPrimals._1)
            debugModel.updateEll(weightedAveragesOfPrimals._2)
          }

          val roundEvaluation =
            if (solverOptions.debug && roundNum % solverOptions.debugMultiplier == 0) {
              // If debug flag is enabled, make few more passes to obtain training error, gap, etc.
              evaluateModel(debugModel, roundNum)
            } else {
              // If debug flag isn't on, perform calculations that don't trigger a shuffle
              val dual = -SolverUtils.objectiveFunction(debugModel.getWeights(), debugModel.getEll(), solverOptions.lambda)
              val elapsedTime = getElapsedTimeSecs()

              RoundEvaluation(roundNum, elapsedTime, 0.0, dual, 0.0, 0.0, 0.0)
            }

          debugSb ++= roundEvaluation + "\n"
      }

    (globalModel, debugSb.toString())
  }

  var gammaZeroInARow = 0;
  def mapper(partitionInfo: (Int, Int), // (partitionIdx, numPartitions)
             dataIterator: Iterator[(Index, InputDataShard[X, Y])],
             helperFunctions: HelperFunctions[X, Y],
             solverOptions: SolverOptions[X, Y],
             localModel: StructSVMModel[X, Y],
             n: Int,
             kAccum: Vector[Int]): Iterator[(Index, ProcessedDataShard[X, Y])] = {

    // println("[Round %d] Beginning mapper at partition %d".format(roundNum, partitionNum))

    val eps: Double = 2.2204E-16

    val maxOracle = helperFunctions.oracleFn
    val phi = helperFunctions.featureFn
    val lossFn = helperFunctions.lossFn

    val lambda = solverOptions.lambda

    val (partitionIdx, numPartitions) = partitionInfo
    var k = kAccum(partitionIdx)

    var ell = localModel.getEll()

    val prevModel = localModel.clone()

    for ((index, shard) <- dataIterator) yield {

      /*if (index < 10)
        println("Partition = %d, Index = %d".format(partitionNum, index))*/

      // 1) Pick example
      val pattern: X = shard.labeledObject.pattern
      val label: Y = shard.labeledObject.label

      // shard.primalInfo: (w_i, ell_i)
      val w_i = shard.primalInfo._1
      val ell_i = shard.primalInfo._2

      // println("w_i is sparse - " + w_i.isInstanceOf[SparseVector[Double]])

      // 2.a) Search for candidates
      val optionalCache_i: Option[BoundedCacheList[Y]] = shard.cache
      val bestCachedCandidateForI: Option[Y] =
        if (solverOptions.enableOracleCache && optionalCache_i.isDefined) {
          val fixedGamma: Double = (2.0 * n) / (k + 2.0 * n)

          val candidates: Seq[(Double, Int)] =
            optionalCache_i.get
              .map(y_i => (((phi(pattern, label) - phi(pattern, y_i)) :* (1 / (n * lambda))),
                (1.0 / n) * lossFn(label, y_i))) // Map each cached y_i to their respective (w_s, ell_s)
              .map {
                case (w_s, ell_s) =>
                  (localModel.getWeights().t * (w_i - w_s) - ((ell_i - ell_s) * (1 / lambda))) /
                    ((w_i - w_s).t * (w_i - w_s) + eps) // Map each (w_s, ell_s) to their respective step-size values 
              }
              .zipWithIndex // We'll need the index later to retrieve the respective approx. ystar_i
              .filter { case (gamma, idx) => gamma > 0.0 }
              .map { case (gamma, idx) => (min(1.0, gamma), idx) } // Clip to [0,1] interval
              .filter { case (gamma, idx) => gamma >= 0.5 * fixedGamma } // Further narrow down cache contenders
              .sortBy { case (gamma, idx) => gamma }

          // If there is a good contender among the cached datapoints, return it
          if (candidates.size >= 1)
            Some(optionalCache_i.get(candidates.head._2))
          else None
        } else None

      // 2.b) Solve loss-augmented inference for point i
      val yAndCache =
        if (bestCachedCandidateForI.isEmpty) {
          val ystar = maxOracle(localModel, pattern, label)

          val updatedCache: Option[BoundedCacheList[Y]] =
            if (solverOptions.enableOracleCache) {

              val nonTruncatedCache =
                if (optionalCache_i.isDefined)
                  optionalCache_i.get :+ ystar
                else
                  MutableList[Y]() :+ ystar

              // Truncate cache to given size and pack it as an Option
              Some(nonTruncatedCache.takeRight(solverOptions.oracleCacheSize))
            } else None

          (ystar, updatedCache)
        } else {
          (bestCachedCandidateForI.get, optionalCache_i)
        }

      val ystar_i = yAndCache._1
      val updatedCache = yAndCache._2

      // 3) Define the update quantities
      val psi_i: Vector[Double] = phi(pattern, label) - phi(pattern, ystar_i) 
      val w_s: Vector[Double] = psi_i :* (1.0 / (n * lambda))
      val loss_i: Double = lossFn(label, ystar_i)
      val ell_s: Double = (1.0 / n) * loss_i

      // 4) Get step-size gamma
      val gamma: Double =
        if (solverOptions.doLineSearch) {
          val thisModel = localModel
          val gamma_opt = (thisModel.getWeights().t * (w_i - w_s) - ((ell_i - ell_s) * (1.0 / lambda))) /
            ((w_i - w_s).t * (w_i - w_s) + eps)
            if(gamma_opt < 0){
               println("[WARNING] gamma_opt < 0 try["+gammaZeroInARow+" of "+n+"]")//TODO remove this statment only works in local mode 
               gammaZeroInARow+=1
            }
            else{
              gammaZeroInARow=0
            }
          if(false){ //TODO //The gamma rule does not hold true if we are no doin excat reconstruction 
             assert(gamma_opt > 0) 
            if( gamma_opt == 0 ){
              println("[WARNING] gamma_opt is zero, are you sure your oracleFn is not violating any assumptions ? ")
            }
          }
          
          max(0.0, min(1.0, gamma_opt))
        } else {
          (2.0 * n) / (k + 2.0 * n)
        }

      val tempWeights1: Vector[Double] = localModel.getWeights() - w_i
      localModel.updateWeights(tempWeights1)
      val w_i_prime = w_i * (1.0 - gamma) + (w_s * gamma)
      val tempWeights2: Vector[Double] = localModel.getWeights() + w_i_prime
      localModel.updateWeights(tempWeights2)

      ell = ell - ell_i
      val ell_i_prime = (ell_i * (1.0 - gamma)) + (ell_s * gamma)
      ell = ell + ell_i_prime

      k += 1

      if (!dataIterator.hasNext) {

        localModel.updateEll(ell)

        val deltaLocalModel = localModel.clone()
        deltaLocalModel.updateWeights(localModel.getWeights() - prevModel.getWeights())
        deltaLocalModel.updateEll(localModel.getEll() - prevModel.getEll())

        val deltaK = k - kAccum(partitionIdx)
        val kAccumLocalDelta = DenseVector.zeros[Int](numPartitions)
        kAccumLocalDelta(partitionIdx) = deltaK

        (index, ProcessedDataShard((w_i_prime - w_i, ell_i_prime - ell_i), updatedCache, Some(LocalSummary(deltaLocalModel, kAccumLocalDelta))))
      } else
        (index, ProcessedDataShard((w_i_prime - w_i, ell_i_prime - ell_i), updatedCache, None))
    }
  }

}