package ch.ethz.dalab.dissolve.optimization

import java.io.File
import java.io.PrintWriter
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.csvwrite
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.regression.LabeledObject
import scala.util.Random
import scala.reflect.ClassTag

/**
 * Train a structured SVM using standard Stochastic (Sub)Gradient Descent (SGD).
 * The implementation here is single machine, not distributed.
 *
 * Input:
 * Each data point (x_i, y_i) is composed of:
 * x_i, the data example
 * y_i, the label
 *
 * @param <X> type for the data examples
 * @param <Y> type for the labels of each example
 */
class SSGSolver[X, Y](
  val indata: Seq[LabeledObject[X, Y]],
  val dissolveFunctions: DissolveFunctions[X, Y],
  val sO: SolverOptions[X, Y]) {

  val roundLimit = sO.roundLimit
  val lambda = sO.lambda
  val debugOn: Boolean = sO.debug

  val maxOracle = dissolveFunctions.oracleFn _
  val phi = dissolveFunctions.featureFn _
  // Number of dimensions of \phi(x, y)
  val ndims: Int = phi(indata(0).pattern, indata(0).label).size


  

  val rand = if(sO.dbcfwSeed==(-1)) new Random() else new Random(sO.dbcfwSeed)
  val data = rand.shuffle(indata).splitAt((indata.length*sO.sampleFrac).asInstanceOf[Int])._1
  val dataSize=data.length
   val testDataSize = if (sO.testData.isDefined) sO.testData.get.length else 0

 val startTime = System.currentTimeMillis()
  def getElapsedTimeSecs(): Double = ((System.currentTimeMillis() - startTime) / 1000.0)
  /**
   * SSG optimizer
   */
  def optimize()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    
   var k: Integer = 0
    val n: Int = data.length
    val d: Int = phi(data(0).pattern, data(0).label).size
    // Use first example to determine dimension of w
    val model: StructSVMModel[X, Y] = new StructSVMModel(DenseVector.zeros(phi(data(0).pattern, data(0).label).size),
      0.0,
      DenseVector.zeros(ndims),
      dissolveFunctions,
      sO.numClasses)

    // Initialization in case of Weighted Averaging
    var wAvg: DenseVector[Double] =
      if (sO.doWeightedAveraging)
        DenseVector.zeros(d)
      else null

    var debugIter = if (sO.debugMultiplier == 0) {
      sO.debugMultiplier = 100
      n
    } else {
      1
    }
    
   
    if (debugOn) {
      println("Beginning training of %d data points in %d passes with lambda=%f".format(n, roundLimit, lambda))
    }

    for (roundNum <- 0 until roundLimit) {

      if (debugOn)
        println("Starting pass #%d".format(roundNum))

      for (dummy <- 0 until n) {
        // 1) Pick example
        val i: Int = dummy
        val pattern: X = data(i).pattern
        val label: Y = data(i).label

        // 2) Solve loss-augmented inference for point i
        val ystar_i: Y = maxOracle(model, pattern, label)

        if(sO.debugWeightUpdate){
        println("#Cmp_Yi#"+","+label.hashCode+","+ystar_i.hashCode)
      }
        
        // 3) Get the subgradient
         val psi_label =phi(pattern, label)
         val psi_yStar_i =  phi(pattern, ystar_i)
         val psi_i: Vector[Double] = psi_label - psi_yStar_i
         val w_s: Vector[Double] = psi_i :* (1 / (n * lambda))
         val w_current =model.getWeights()
        // 4) Step size gamma
        val gamma: Double = 1.0 / (k + 1.0)

        // 5) Update the weights of the model
        val newWeights: Vector[Double] = (w_current :* (1 - gamma)) + (w_s :* (gamma * n))
        model.updateWeights(newWeights)

        k = k + 1

        if (sO.doWeightedAveraging) {
          val rho: Double = 2.0 / (k + 2.0)
          wAvg = wAvg * (1.0 - rho) + model.getWeights() * rho
        }

        if(sO.debugWeightUpdate){
        if(k==0){
          println("#UpdateLog#,vect,k,"+(0 until psi_i.size).toArray.mkString(",c"))
        }
        println("#UpdateLog#,Psi_label,"+k+","+psi_label.toArray.mkString("\t,"))
        println("#UpdateLog#,Psi_yStar_i,"+k+","+psi_yStar_i.toArray.mkString("\t,"))
        println("#UpdateLog#,Psi_i,"+k+","+psi_i.toArray.mkString("\t,"))
        println("#UpdateLog#,W_cur,"+k+","+w_current.toArray.mkString("\t,"))
        println("#UpdateLog#,W_new,"+k+","+model.getWeights().toArray.mkString("\t,"))
        println("#UpdateLog#,W_s,"+k+","+w_s.toArray.mkString("\t,"))
 
      }
        
      
        
        
        
        /*if (debugOn && k >= debugIter) {
          if (solverOptions.doWeightedAveraging) {
            debugModel.updateWeights(wAvg)
          } else {
            debugModel.updateWeights(model.getWeights)
          }

          val primal = SolverUtils.primalObjective(data, featureFn, lossFn, oracleFn, debugModel, lambda)
          val trainError = SolverUtils.averageLoss(data, lossFn, predictFn, debugModel)

          if (solverOptions.testData != null) {
            val testError = SolverUtils.averageLoss(solverOptions.testData, lossFn, predictFn, debugModel)
            println("Pass %d Iteration %d, SVM primal = %f, Train error = %f, Test error = %f"
              .format(passNum + 1, k, primal, trainError, testError))

            if (solverOptions.debug)
              lossWriter.write("%d,%d,%f,%f,%f\n".format(passNum + 1, k, primal, trainError, testError))
          } else {
            println("Pass %d Iteration %d, SVM primal = %f, Train error = %f"
              .format(passNum + 1, k, primal, trainError))
            if (solverOptions.debug)
              lossWriter.write("%d,%d,%f,%f,\n".format(passNum + 1, k, primal, trainError))
          }

          debugIter = min(debugIter + n, ceil(debugIter * (1 + solverOptions.debugMultiplier / 100)))
        }*/

      }
      
   if (sO.debug && roundNum % sO.debugMultiplier == 0) {
              // If debug flag is enabled, make few more passes to obtain training error, gap, etc.
     val debugModel: StructSVMModel[X, Y] = model.clone()

              evaluateModel(debugModel, roundNum)
         }
      if (debugOn)
        println("Completed pass #%d".format(roundNum))

    }

    return model
  }
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
   
   def evaluateModel(model: StructSVMModel[X, Y], roundNum: Int = 0)(implicit m: ClassTag[Y]): RoundEvaluation = {
      val dual = -SolverUtils.objectiveFunction(model.getWeights(), model.getEll(), sO.lambda)
      val dualityGap = SolverUtils.dualityGap(data, dissolveFunctions, model, sO.lambda, dataSize)._1
      val primal = dual + dualityGap

      val trainError = SolverUtils.averageLoss(data, dissolveFunctions, model, dataSize)
      val testError =
        if (sO.testData.isDefined)
          SolverUtils.averageLoss(sO.testData.get, dissolveFunctions, model, testDataSize)
        else
          0.00

      val elapsedTime = getElapsedTimeSecs()

      println("[%.3f] Round = %d, Gap = %f, Primal = %f, Dual = %f, TrainLoss = %1.5e, TestLoss = %1.5e"
        .format(elapsedTime, roundNum, dualityGap, primal, dual, trainError, testError))
      
        
          //assert(dualityGap>0)
      def bToS( a:Boolean)={if(a)"t"else"f"}
      
      def w_Norm():Double={
       val w = DenseVector(model.weights.toArray) 
       norm(w) 
      }
      def w_unaryNorm():Double={
          val w = DenseVector(model.weights.toArray) 
        val pairwiseWlength = sO.numClasses*sO.numClasses* (if(sO.modelPairwiseDataDependent) sO.numDataDepGraidBins else 1)
        val unaryLeng = w.length-pairwiseWlength
        val unaryW=w(0 until unaryLeng)
        assert(unaryW.length == unaryLeng)
        norm(unaryW)
      }
      def w_pairWiseNorm():Double={
        val w = DenseVector(model.weights.toArray) 
        val pairwiseWlength = sO.numClasses*sO.numClasses* (if(sO.modelPairwiseDataDependent) sO.numDataDepGraidBins else 1)
        val unaryLeng = w.length-pairwiseWlength
        val pairW=w(unaryLeng to -1)
        assert(pairW.length == pairwiseWlength)
        norm(pairW)
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
            ","+sO.lambda+","+bToS(sO.standardizeFeaturesByColumn)+","+bToS(sO.featUniqueIntensity)+","+bToS(sO.featAddSupSize)+","+sO.slicMinBlobSize+","+bToS(sO.optimizeWithSubGraid))
       
   
        
    val a=0
        
      RoundEvaluation(roundNum, elapsedTime, primal, dual, dualityGap, trainError, testError)
    }

}