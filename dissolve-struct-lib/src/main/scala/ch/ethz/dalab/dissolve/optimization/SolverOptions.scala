package ch.ethz.dalab.dissolve.optimization

import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg.Vector
import org.apache.spark.rdd.RDD
import java.io.File

sealed trait StoppingCriterion

// Option A - Limit number of communication rounds
case object RoundLimitCriterion extends StoppingCriterion {
  override def toString(): String = { "RoundLimitCriterion" }
}

// Option B - Check gap
case object GapThresholdCriterion extends StoppingCriterion {
  override def toString(): String = { "GapThresholdCriterion" }
}

// Option C - Run for this amount of time (in secs)
case object TimeLimitCriterion extends StoppingCriterion {
  override def toString(): String = { "TimeLimitCriterion" }
}

class SolverOptions[X, Y] extends Serializable {
  var doWeightedAveraging: Boolean = false

  var randSeed: Int = 42
  var dbcfwSeed: Int = -1;
  /**
   *  BCFW - "uniform", "perm" or "iter"
   *  DBCFW - "count", "frac"
   */
  var gitVersion = "Null"
  var runName = "UnNamed"
  var onlyUnary = false
  var startTime = 0L //
  var sample: String = "frac"
  var lambda: Double = 0.01 // FIXME This is 1/n in Matlab code
  
  var testData: Option[Seq[LabeledObject[X, Y]]] = Option.empty[Seq[LabeledObject[X, Y]]]
  var testDataRDD: Option[RDD[LabeledObject[X, Y]]] = Option.empty[RDD[LabeledObject[X, Y]]]

  var doLineSearch: Boolean = true

  // Checkpoint once in these many rounds
  var checkpointFreq: Int = 50

  // In case of multi-class
  var numClasses = -1
  
  //Synthetic Data Generator 
  var dataWasGenerated = false
  var dataGenSparsity = 0.0
  var dataAddedNoise = 0.0
  var dataNoiseOnlyTest = false
  var dataGenTestSize = 30
  var dataGenTrainSize = 30
  var dataGenCanvasSize=16
  var dataRandSeed= (-1)
  var isColor = true
  var dataSetName ="NONAME"
  var inferenceMethod="NONAME"
  var useClassFreqWeighting = false
  var initWithEmpiricalTransProb= false
  var initWeight= Array.fill(0){0.0}
  var LOSS_AUGMENTATION_OVERRIDE = false
  var putLabelIntoFeat = false //defunc
  var useMSRC = false //defunc
  var generateMSRCSupPix = false //defunc
  var squareSLICoption=false
  var useNaiveUnaryMax = false
  var trainTestEqual=false
  var superPixelSize=(-1)
  var dataFilesDir=""
  var imageDataFilesDir=""
  var groundTruthDataFilesDir=""
  var PAIRWISE_UPPER_TRI=true
  var dataGenSquareSize=10
  var dataGenSquareNoise=0.0
  var dataGenHowMany=40
  var dataGenOsilNoise=0.0
  var slicCompactness = 5.0 
  var modelPairwiseDataDependent=false
  var featIncludeMeanIntensity=false
  //Which decode func 
  var useMF=false
  var learningRate = 0.1
  var mfTemp = 5.0
  var weighDownPairwise = 1.0
  var weighDownUnary = 1.0
  // Cache params
  var enableOracleCache: Boolean = false
  var oracleCacheSize: Int = 10
  var dataGenGreyOnly:Boolean = false
  var compPerPixLoss:Boolean=false
  var dataGenEnforNeigh:Boolean=true
  var dataGenNeighProb:Double =1.0
  var debugPrintSuperPixImg:Boolean=false
  var featHistSize:Int=4
  var featCoOcurNumBins:Int=3
  var useLoopyBP:Boolean=false
  var useMPLP:Boolean=false
  var slicNormalizePerClust:Boolean=true
  var featAddOffsetColumn:Boolean=false
  var featAddIntensityVariance:Boolean=false
  
  var recompFeat:Boolean=false
  var featUniqueIntensity:Boolean=false
  var featUnique2Hop:Boolean=false
  var maxColorValue:Int=255
  var dataDepUseIntensity:Boolean=true
  var dataDepUseIntensityByNeighSD:Boolean=false
  var dataDepUseIntensityBy2NeighSD:Boolean=false
  var dataDepUseUniqueness:Boolean=false
  var dataDepUseUniquenessInOtherNeighbourhood:Boolean=false
  var dataDepMeth:String = "dataDepUseIntensity"
    
  var slicMinBlobSize:Int=(-1)
  var standardizeFeaturesByColumn:Boolean=false
  var featNeighHist:Boolean=false
  var preStandardizeImagesFirst:Boolean = false
  var featUseStdHist:Boolean=false
  
  
  var globalMean:Double = Double.MinValue
  var globalVar:Double = Double.MinValue
  
  var loopyBPmaxIter:Int=10;
  var numDataDepGraidBins:Int=5;
  var alsoWeighLossAugByFreq:Boolean=true;
  var splitImagesBy:Int=(-1);
  var optimizeWithSubGraid:Boolean = false;
  var featAddSupSize:Boolean = false;
  var pairwiseModelPruneSomeEdges:Double = 0.0;
  var useRandomDecoding:Boolean = false; 
  var slicSimpleEdgeFinder:Boolean=false;
  var filterOutImagesWithOnlyOneLabel:Boolean=false;
  var leaveOneOutCrossVal:Boolean=false;
  var curLeaveOutIteration:Int = -1;
  var leaveOutCVmaxIter:Int = Integer.MAX_VALUE
  var numberOfCoresToUse:Int = -1;
  var logOracleTiming:Boolean = false;
  var spark_driver_memory:String = "";
  
  
  
  
  
  // DBCFW specific params
  var H: Int = 5 // Number of data points to sample in each round of CoCoA (= number of local coordinate updates)
  var sampleFrac: Double = 0.5
  var sampleWithReplacement: Boolean = false

  var enableManualPartitionSize: Boolean = false
  var NUM_PART: Int = 1 // Number of partitions of the RDD

  // For debugging/Testing purposes
  // Basic debugging flag
  var debug: Boolean = false
  var debugWeightUpdate:Boolean = false
  // Obtain statistics (primal value, duality gap, train error, test error, etc.) once in these many rounds.
  // If 1, obtains statistics in each round
  var debugMultiplier: Int = 1

  // Option A - Limit number of communication rounds
  var roundLimit: Int = 25

  // Option B - Check gap
  var gapThreshold: Double = 0.1
  var gapCheck: Int = 1 // Check for once these many rounds

  // Option C - Run for this amount of time (in secs)
  var timeLimit: Int = 300

  var stoppingCriterion: StoppingCriterion = RoundLimitCriterion

  // Sparse representation of w_i's
  var sparse: Boolean = false

  // Path to write the CSVs
  var debugInfoPath: String = new File(".").getCanonicalPath() + "/debugInfo-%d.csv".format(System.currentTimeMillis())

  override def toString(): String = {
    val sb: StringBuilder = new StringBuilder()

    sb ++= "# numRounds=%s\n".format(roundLimit)
    sb ++= "# doWeightedAveraging=%s\n".format(doWeightedAveraging)

    sb ++= "# randSeed=%d\n".format(randSeed)

    sb ++= "# sample=%s\n".format(sample)
    sb ++= "# lambda=%f\n".format(lambda)
    sb ++= "# doLineSearch=%s\n".format(doLineSearch)

    sb ++= "# enableManualPartitionSize=%s\n".format(enableManualPartitionSize)
    sb ++= "# NUM_PART=%s\n".format(NUM_PART)

    sb ++= "# enableOracleCache=%s\n".format(enableOracleCache)
    sb ++= "# oracleCacheSize=%d\n".format(oracleCacheSize)

    sb ++= "# H=%d\n".format(H)
    sb ++= "# sampleFrac=%f\n".format(sampleFrac)
    sb ++= "# sampleWithReplacement=%s\n".format(sampleWithReplacement)

    sb ++= "# debugInfoPath=%s\n".format(debugInfoPath)

    sb ++= "# checkpointFreq=%d\n".format(checkpointFreq)

    sb ++= "# stoppingCriterion=%s\n".format(stoppingCriterion)
    this.stoppingCriterion match {
      case RoundLimitCriterion   => sb ++= "# roundLimit=%d\n".format(roundLimit)
      case GapThresholdCriterion => sb ++= "# gapThreshold=%f\n".format(gapThreshold)
      case TimeLimitCriterion    => sb ++= "# timeLimit=%d\n".format(timeLimit)
      case _                     => throw new Exception("Unrecognized Stopping Criterion")
    }

    sb ++= "# debugMultiplier=%d\n".format(debugMultiplier)

    sb.toString()
  }

}