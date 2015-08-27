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
import breeze.linalg._
import breeze.numerics._
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
  val sO: SolverOptions[X, Y],
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
    val testDataSize = if (sO.testDataRDD.isDefined) sO.testDataRDD.get.count().toInt else 0

    val verboseDebug: Boolean = false

    val d: Int = dissolveFunctions.featureFn(samplePoint.pattern, samplePoint.label).size
    // Let the initial model contain zeros for all weights
    // Global model uses Dense Vectors by default
    
    if(sO.initWithEmpiricalTransProb){
      
      val tmp =DenseVector(sO.initWeight)
      print("##Using Init Weight: "+tmp)
    }
    
    
    var globalModel: StructSVMModel[X, Y] = new StructSVMModel[X, Y]( if(sO.initWithEmpiricalTransProb) DenseVector(sO.initWeight) else DenseVector.zeros(d), 0.0,
      DenseVector.zeros(d), dissolveFunctions, sO.numClasses)

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
        if (sO.sparse) // w_i can be either Sparse or Dense 
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
      if (sO.enableOracleCache)
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
      if (sO.sample == "frac")
        sO.sampleFrac
      else if (sO.sample == "count")
        math.min(sO.H / dataSize, 1.0)
      else {
        println("[WARNING] %s is not a valid option. Reverting to sampleFrac = 0.5".format(sO.sample))
        0.5
      }
    }

    /**
     * In case of weighted averaging, start off with an all-zero (wAvg, lAvg)
     */
    var wAvg: Vector[Double] =
      if (sO.doWeightedAveraging)
        DenseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    var weightedAveragesOfPrimals: PrimalInfo =
      if (sO.doWeightedAveraging)
        (DenseVector.zeros(d), 0.0)
      else null

    var iterCount: Int = 0

    def getLatestModel(): StructSVMModel[X, Y] = {
      val debugModel: StructSVMModel[X, Y] = globalModel.clone()
      if (sO.doWeightedAveraging) {
        debugModel.updateWeights(weightedAveragesOfPrimals._1)
        debugModel.updateEll(weightedAveragesOfPrimals._2)
      }
      debugModel
    }

    def getLatestGap(): Double = {
      val debugModel: StructSVMModel[X, Y] = getLatestModel()
      val gap = SolverUtils.dualityGap(data, dissolveFunctions, debugModel, sO.lambda, dataSize)
      gap._1
    }

    def evaluateModel(model: StructSVMModel[X, Y], roundNum: Int = 0): RoundEvaluation = {
      val dual = -SolverUtils.objectiveFunction(model.getWeights(), model.getEll(), sO.lambda)
      val dualityGap = SolverUtils.dualityGap(data, dissolveFunctions, model, sO.lambda, dataSize)._1
      val primal = dual + dualityGap

      val trainError = SolverUtils.averageLoss(data, dissolveFunctions, model, dataSize)
      val testError =
        if (sO.testDataRDD.isDefined)
          SolverUtils.averageLoss(sO.testDataRDD.get, dissolveFunctions, model, testDataSize)
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
      
      def w_Norm():Double={
       val w = DenseVector(model.weights.toArray) 
       norm(w) 
      }
     def w_unaryNorm():Double={
          val w = DenseVector(model.weights.toArray) 
        val pairwiseWlength = if(sO.onlyUnary ) 0 else sO.numClasses*sO.numClasses* (if(sO.modelPairwiseDataDependent) sO.numDataDepGraidBins else 1)
        val unaryLeng = w.length-pairwiseWlength
        val unaryW=w(0 until unaryLeng)
        assert(unaryW.length == unaryLeng)
        norm(unaryW)
      }
      def w_pairWiseNorm():Double={
        if(sO.onlyUnary )
          -1.0
        else{
        val w = DenseVector(model.weights.toArray) 
        val pairwiseWlength = sO.numClasses*sO.numClasses* (if(sO.modelPairwiseDataDependent) sO.numDataDepGraidBins else 1)
        val unaryLeng = w.length-pairwiseWlength
        val pairW=w(unaryLeng to -1)
        assert(pairW.length == pairwiseWlength)
        norm(pairW)
        }
      }
      def w_maxPairWiseNorm():(Int,Double,Int,Double)={
        if(sO.modelPairwiseDataDependent){
        val weightVec = DenseVector(model.weights.toArray) 
        val pairwiseWlength = sO.numClasses*sO.numClasses*sO.numDataDepGraidBins
        val unaryLeng = weightVec.length-pairwiseWlength
        val pairW=weightVec(unaryLeng to -1)
            
        // Pairwise feature Vector
        val pairwiseSize=sO.numClasses*sO.numClasses
        val pairwiseMats = Array.fill(sO.numDataDepGraidBins){DenseMatrix.zeros[Double](sO.numClasses,sO.numClasses)}
     
        val unaryEnd = unaryLeng
        val allNorms=for(i <- 0 until sO.numDataDepGraidBins) yield {
          val startI = unaryEnd + ( i*pairwiseSize)
          val endI = startI + pairwiseSize
          val pairwiseFeatureVec = weightVec(startI until endI).toDenseVector
           assert(pairwiseFeatureVec.size == sO.numClasses * sO.numClasses, "was ="+pairwiseFeatureVec.size  +" should have been= "+(sO.numClasses * sO.numClasses))
           pairwiseMats(i)=pairwiseFeatureVec.toDenseMatrix.reshape(sO.numClasses, sO.numClasses)
           
                          
            val thetaPairwise=pairwiseMats(i)
            val curNorm =norm(pairwiseFeatureVec)
            println("------------- Pairwise_Mat %d -------------  Norm( %1.3e )".format(i,curNorm))
            print("Diagonal: [[")
            for(i <- 0 until thetaPairwise.rows ){
              print("\t,%1.3e".format(thetaPairwise(i,i)))
            }
            print("]]\n")
            
              for(r<- 0 until thetaPairwise.rows ){
                for( c<- 0 until thetaPairwise.cols){
                print("\t,%1.3e".format(thetaPairwise(r,c)))
              }
                print("\n")
            }
 
           curNorm
        }
        val normsDV = DenseVector(allNorms.toArray)
        
        (argmax(normsDV),max(normsDV),argmin(normsDV),min(normsDV))
        }
        else{
          ((-1),(-1.0),(-1),(-1.0))
        }

      }
      val (whichDataDepWasMaxWNorm,dataDepWasMaxWNorm,whichDataDepWasMinWNorm,dataDepWasMinWNorm) =w_maxPairWiseNorm
      
      val newStats = " %1.3e, %1.3e, %1.3e, %d, %1.3e, %d, %1.3e,".format(w_Norm,w_unaryNorm,w_pairWiseNorm,whichDataDepWasMaxWNorm,dataDepWasMaxWNorm,whichDataDepWasMinWNorm,dataDepWasMinWNorm)
      
        println("#RoundProgTag# ,%d, %s , %s , %.3f, %d, %.6f, %.6f, %.6f, %.6f, %.6f , %.2f, %s, %s, %s, %d, %s, %s, %s, %d, %s, %s, %.3f, %d, %d, %s, %s, %s, %.3f, %d, %d, %.3f, %s, %.3f"
        .format(sO.startTime, sO.runName,sO.gitVersion,elapsedTime, roundNum, dualityGap, primal,
            dual, trainError, testError,sO.sampleFrac, if(sO.doWeightedAveraging) "t" else "f", 
            if(sO.onlyUnary) "t" else "f" ,if(sO.squareSLICoption) "t" else "f" , sO.superPixelSize, sO.dataSetName, if(sO.trainTestEqual)"t" else "f",
            sO.inferenceMethod,sO.dbcfwSeed, if(sO.dataGenGreyOnly) "t" else "f", if(sO.compPerPixLoss) "t" else "f", sO.dataGenNeighProb, sO.featHistSize,
            sO.featCoOcurNumBins, if(sO.useLoopyBP) "t" else "f", if(sO.useMPLP) "t" else "f", bToS(sO.slicNormalizePerClust), sO.dataGenOsilNoise, sO.dataRandSeed,
            sO.dataGenHowMany,sO.slicCompactness,bToS(sO.putLabelIntoFeat),sO.dataAddedNoise
            )+","+(if(sO.modelPairwiseDataDependent) "t" else "f")+","+(if(sO.featIncludeMeanIntensity) "t" else "f")+","+bToS(sO.featAddOffsetColumn)+
            ","+bToS(sO.featAddIntensityVariance)+","+bToS(sO.featNeighHist)+","+ sO.numDataDepGraidBins+","+sO.loopyBPmaxIter+","+newStats+sO.dataDepMeth+","+model.weights.length+
            ","+sO.lambda+","+bToS(sO.standardizeFeaturesByColumn)+","+bToS(sO.featUniqueIntensity)+","+bToS(sO.featAddSupSize)+","+sO.slicMinBlobSize+","+bToS(sO.optimizeWithSubGraid)+
            ","+sO.curLeaveOutIteration+","+sO.numberOfCoresToUse+","+sO.NUM_PART)
       
   
        
    val a=0
        
      RoundEvaluation(roundNum, elapsedTime, primal, dual, dualityGap, trainError, testError)
    }

    println("Beginning training of %d data points in %d passes with lambda=%f".format(dataSize, sO.roundLimit, sO.lambda))

    debugSb ++= "round,time,primal,dual,gap,train_error,test_error\n"

    def getElapsedTimeSecs(): Double = ((System.currentTimeMillis() - startTime) / 1000.0)

    /**
     * ==== Begin Training rounds ====
     */
    Stream.from(1)
      .takeWhile {
        roundNum =>
          val continueExecution =
            sO.stoppingCriterion match {
              case RoundLimitCriterion => roundNum <= sO.roundLimit
              case TimeLimitCriterion  => getElapsedTimeSecs() < sO.timeLimit
              case GapThresholdCriterion =>
                // Calculating duality gap is really expensive. So, check ever gapCheck rounds
                if (roundNum % sO.gapCheck == 0)
                  getLatestGap() > sO.gapThreshold
                else
                  true
              case _ => throw new Exception("Unrecognized Stopping Criterion")
            }

          if (sO.debug && (!(continueExecution || (roundNum - 1 % sO.debugMultiplier == 0)) || roundNum == 1)) {
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
          
          val indexedJointData: RDD[(Index, InputDataShard[X, Y])] = if(sO.dbcfwSeed==(-1) )
            indexedTrainDataRDD
              .sample(sO.sampleWithReplacement, sampleFrac)   
              .join(indexedPrimalsRDD)
              .leftOuterJoin(indexedCacheRDD)
              .mapValues { // Because mapValues preserves partitioning
                case ((labeledObject, primalInfo), cache) =>
                  InputDataShard(labeledObject, primalInfo, cache)
              }
          else
indexedTrainDataRDD
              .sample(sO.sampleWithReplacement, sampleFrac,sO.dbcfwSeed)  
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
                  sO,
                  globalModel,
                  dataSize,
                  kAccum),
              preservesPartitioning = true)
              .cache()

          /**
           * Step 2.5 - A long lineage may cause a StackOverFlow error in the JVM.
           * So, trigger a checkpointing once in a while.
           */
          if (roundNum % sO.checkpointFreq == 0) {
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
          if (sO.doWeightedAveraging) {
            debugModel.updateWeights(weightedAveragesOfPrimals._1)
            debugModel.updateEll(weightedAveragesOfPrimals._2)
          }

          val roundEvaluation =
            if (sO.debug && roundNum % sO.debugMultiplier == 0) {
              // If debug flag is enabled, make few more passes to obtain training error, gap, etc.
              evaluateModel(debugModel, roundNum)
            } else {
              // If debug flag isn't on, perform calculations that don't trigger a shuffle
              val dual = -SolverUtils.objectiveFunction(debugModel.getWeights(), debugModel.getEll(), sO.lambda)
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
      val psi_label =phi(pattern, label)
      val psi_yStar_i =  phi(pattern, ystar_i)
      val psi_i: Vector[Double] = psi_label - psi_yStar_i
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
          if(solverOptions.debugWeightUpdate){
            println("#OptmizeLog#,%d , %1.3e , %1.5e , %1.5e , %1.5e , %1.5e ".format(k,gamma_opt,max(0.0, min(1.0, gamma_opt)),loss_i,ell_s,norm(DenseVector(psi_i.toArray))))
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

      
      if(solverOptions.debugWeightUpdate){
        if(k==0){
          println("#UpdateLog#,vect,k,"+(0 until psi_i.size).toArray.mkString(",c"))
        }
        println("#UpdateLog#,Psi_label,"+k+","+psi_label.toArray.mkString(","))
        println("#UpdateLog#,Psi_yStar_i,"+k+","+psi_yStar_i.toArray.mkString(","))
        println("#UpdateLog#,Psi_i,"+k+","+psi_i.toArray.mkString(","))
        println("#UpdateLog#,W_localmodel,"+k+","+localModel.getWeights().toArray.mkString(","))
        println("#UpdateLog#,W_s,"+k+","+w_s.toArray.mkString(","))
        println("#UpdateLog#,W_i,"+k+","+w_i.toArray.mkString(","))
        println("#UpdateLog#,W_i_prime,"+k+","+w_i_prime.toArray.mkString(","))
        
      }
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