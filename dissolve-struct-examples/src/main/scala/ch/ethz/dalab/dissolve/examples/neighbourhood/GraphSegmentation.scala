package ch.ethz.dalab.dissolve.examples.neighbourhood

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
import scala.collection.mutable.HashSet

class GraphSegmentationClass(DISABLE_PAIRWISE:Boolean, MAX_DECODE_ITERATIONS:Int, MF_LEARNING_RATE:Double=0.1, USE_MF:Boolean=false, MF_TEMP:Double=5.0,USE_NAIV_UNARY_MAX:Boolean=false, DEBUG_COMPARE_MF_FACTORIE:Boolean=false, MAX_DECODE_ITERATIONS_MF_ALT:Int, EXP_NAME:String="NoName") extends DissolveFunctions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] with Serializable {
    
  type xData = GraphStruct[Vector[Double], (Int, Int, Int)]
  type yLabels = GraphLabels

  println("GraphSegmentation::: (DISABLE_PAIRWISE= " + DISABLE_PAIRWISE + " )")
  //Convert the graph into one big feature vector 
  def featureFn(xDat: xData, yDat: yLabels): Vector[Double] = {
    assert(xDat.graphNodes.size == yDat.d.size)
    

    val xFeatures = xDat.getF(0).size
    val numClasses = yDat.numClasses

    val unaryFeatureSize = xFeatures * numClasses
    val pairwiseFeatureSize = numClasses * numClasses
    val phi = if (!DISABLE_PAIRWISE) DenseVector.zeros[Double](unaryFeatureSize + pairwiseFeatureSize) else DenseVector.zeros[Double](unaryFeatureSize)

    val unary = DenseVector.zeros[Double](unaryFeatureSize)
    for (idx <- 0 until yDat.d.size) {
      val label = yDat.d(idx)
      val startIdx = label * xFeatures
      val endIdx = startIdx + xFeatures
      val curF =xDat.getF(idx) 
      if(curF.size!=(endIdx-startIdx))
        print("wtf")
      if(endIdx > unary.size)
        print("wtf2")
      unary(startIdx until endIdx) :=curF + unary(startIdx until endIdx)
    }

    phi(0 until (unaryFeatureSize)) := unary

    if (!DISABLE_PAIRWISE) {
      val pairwise = normalize(getPairwiseFeatureMap(yDat, xDat).toDenseVector) //TODO does this toDenseVector actually use proper columnIndex form, or atleast is it deterministic ? 
      assert((phi.size - unaryFeatureSize) == pairwise.size)
      phi((unaryFeatureSize) until phi.size) := pairwise

    }

   phi
  }

  // Count pairwise occurances of classes. This is normalized on the outside 
  //TODO why do we recount this every round. cant we just cache it somewhere
  def getPairwiseFeatureMap(yi: yLabels, xi: xData): DenseMatrix[Double] = {

    val pairwiseMat = DenseMatrix.zeros[Double](yi.numClasses, yi.numClasses)
    for (idx <- 0 until xi.size) {
      val myLabel = yi.d(idx)
      xi.getC(idx).foreach { neighbour => { pairwiseMat(myLabel, yi.d(neighbour)) += 1 } }
    }
    pairwiseMat
  }

  def lossFn(yTruth: yLabels, yPredict: yLabels): Double = {
    assert(yPredict.d.size == yTruth.d.size)
    //val classFreqs = yTruth.classFreq

    val loss =
      for (
        idx <- 0 until yPredict.d.size
      ) yield {
        //TODO put this back to weighted loss 
        //if (yTruth.get(x, y, z) == yPredict.get(x, y, z)) 0.0 else 1.0 / classFreqs.get(yTruth.get(x, y, z)).get // Insert classFrequency back into the truthObject
        if (yTruth.d(idx) == yPredict.d(idx)) 0.0 else 1.0 // Insert classFrequency back into the truthObject
      }

    loss.sum / (yPredict.d.size)
  }

  def xFeatureStack(xi: xData): DenseMatrix[Double] = {
    val featureStack = DenseMatrix.zeros[Double](xi.getF(0).size, xi.size)
    for (i <- 0 until xi.size) {
      featureStack(::, i) := xi.getF(i)
    }
    featureStack
  }

  def decodeFn(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], graph: xData, debug: Boolean = false): yLabels = {

    val t0 = System.currentTimeMillis()
    val model = new ItemizedModel
    val numRegions: Int = thetaUnary.rows
    assert(numRegions == graph.size)
    val numClasses: Int = thetaUnary.cols

    class RegionVar(val score: Int) extends IntegerVariable(score)

    object PixelDomain extends DiscreteDomain(numClasses)

    class Pixel(i: Int) extends DiscreteVariable(i) { //i is just the initial value 
      def domain = PixelDomain
    }

    val labelParams = Array.fill(graph.size) { new Pixel(0) }
    val nodePairsUsed: HashSet[(Int, Int)] = new HashSet[(Int, Int)]()
    for (idx <- 0 until labelParams.length) {
      model ++= new Factor1(labelParams(idx)) {
        def score(k: Pixel#Value) = thetaUnary(idx, k.intValue)
      }

      if (!DISABLE_PAIRWISE) {

        graph.getC(idx).foreach { neighbour =>
          {
            if (!nodePairsUsed.contains((idx, neighbour)) && !nodePairsUsed.contains((neighbour, idx))) { //This prevents adding neighbours twice 
              nodePairsUsed.add((idx, neighbour))
              model ++= new Factor2(labelParams(idx), labelParams(neighbour)) {
                def score(i: Pixel#Value, j: Pixel#Value) = thetaPairwise(i.intValue, j.intValue)
              }
            }
          }
        }
      }
    }

     // println("nodePairsFound" +nodePairsUsed.size+" Input thetaUnary("+thetaUnary.rows+","+thetaUnary.cols+")/nFactor Graph Size: "+model.factors.size)//TODO remove 
    val maxIterations = MAX_DECODE_ITERATIONS
    val maximizer = new MaximizeByMPLP(maxIterations)
    val assgn = maximizer.infer(labelParams, model).mapAssignment //Where the magic happens 
    // Retrieve assigned labels from these pixels
    val out = Array.fill[Int](graph.size)(0)
    for (i <- 0 until out.size) {
      out(i) = assgn(labelParams(i)).intValue
    }
    val t1 = System.currentTimeMillis()
    //print(" decodeTime=[%d s]".format(  (t1-t0)/1000  ))
    new GraphLabels(Vector(out), numClasses)
  }

  var counter =0; //TODO REMOVE 
  var lastHash:Int=0;
  def oracleFn(model: StructSVMModel[xData, yLabels], xi: xData, yi: yLabels): yLabels = {
    val thisyiHash= if(yi!=null) yi.hashCode else 0
    val numClasses = model.numClasses

    val numDims = xi.getF(0).length

    val weightVec = model.getWeights()

    // Unary is of size f x K, each column representing feature vector of class K
    // Pairwise if of size K * K
    val (unaryWeights, pairwiseWeights) = unpackWeightVec(weightVec, numDims, numClasses = numClasses, padded = false)
    assert(unaryWeights.rows == numDims)
    assert(unaryWeights.cols == numClasses)
    assert(DISABLE_PAIRWISE || pairwiseWeights.rows == pairwiseWeights.cols)
    assert(DISABLE_PAIRWISE || pairwiseWeights.rows == numClasses)

    val phi_Y: DenseMatrix[Double] = xFeatureStack(xi) // Retrieves a f x r matrix representation of the original image, i.e, each column is the feature vector that region r
    val thetaUnary = phi_Y.t * unaryWeights // Returns a r x K matrix, where theta(r, k) is the unary potential of region r having label k
    //Aureliens code calls this Unary potential (without the loss, but its ok) 
    
    //TODO check if this transpose makes sense 

    val thetaPairwise = pairwiseWeights

    // If yi is present, do loss-augmentation
    if (yi != null) {
      for (idx <- 0 until xi.size) { //TODO check if this is using correct indexs 
        thetaUnary(idx, ::) := thetaUnary(idx, ::) + 1.0 / xi.size //We are using a zero-one loss per y so here there are just constants
        // Loss augmentation step
        val k = yi.d(idx)
        thetaUnary(idx, k) = thetaUnary(idx, k) - 1.0 / xi.size //This is a zero loss b/c it cancels out the +1 for all non correct labels 
        //This zero one loss is repeated code from the lossFn. lossFn gets loss for  
        //     a whole image but inside it uses zero-one loss for pixel comparison 
        //     this should be a function so it is common between the two uses 

      }

    }

    
    
    /**
     * Parameter estimation
     */
    val startTime = System.currentTimeMillis()
   
    
    
    //TODO REMOVE DEBUG 
    
    /*
    val t1_1 =System.currentTimeMillis();
    val deco1 = MeanFieldTest.decodeFn_AR(thetaUnary,thetaPairwise,graph=xi, maxIterations  = MAX_DECODE_ITERATIONS,temp=MF_TEMP, debug= false ,logTag=thisyiHash.toString())
    val t1_2 =System.currentTimeMillis();
    val t2_1 =System.currentTimeMillis();
    val deco2 = MeanFieldTest.decodeFn_PR(thetaUnary,thetaPairwise,graph=xi, maxIterations  = MAX_DECODE_ITERATIONS,temp=MF_TEMP, debug= false ,logTag=thisyiHash.toString())
    val t2_2 =System.currentTimeMillis();
    println("#MF_time:"+(t1_2-t1_1)+" MF_parTime:"+(t2_2-t2_1)+(if(deco1.equals(deco2))"and ARE SAME"else"and ARE DIF"));
    */
    
    
    val decoded =   if(USE_MF){
      MeanFieldTest.decodeFn_PR(thetaUnary,thetaPairwise,graph=xi, maxIterations  = MAX_DECODE_ITERATIONS_MF_ALT,temp=MF_TEMP, debug= false ,logTag=thisyiHash.toString())
      }
    else if(USE_NAIV_UNARY_MAX){
      NaiveUnaryMax.decodeFn(thetaUnary)
    }
    else{
      decodeFn(thetaUnary, thetaPairwise, xi, debug = false)
      }
    
    if(DEBUG_COMPARE_MF_FACTORIE){
      val factorieDecode = decodeFn(thetaUnary, thetaPairwise, xi, debug = false)
      val mfDecode =  MeanFieldTest.decodeFn_PR(thetaUnary,thetaPairwise,graph=xi, maxIterations  = MAX_DECODE_ITERATIONS_MF_ALT,temp=MF_TEMP, debug= false ,logTag=thisyiHash.toString())
      val naiveDecode =  NaiveUnaryMax.decodeFn(thetaUnary)
      
      val halfEfact  = halfEnergyOf(factorieDecode,thetaUnary,thetaPairwise,xi)
      val halfEmf = halfEnergyOf(mfDecode,thetaUnary,thetaPairwise,xi)
      val halfEnaive = halfEnergyOf(naiveDecode,thetaUnary,thetaPairwise,xi)
      println("#DecEngCmp#,%d,%d,%.5f,%.5f,%.5f,%d,%s".format( (if(yi!=null) thisyiHash else -1 ),System.currentTimeMillis(),halfEfact,halfEmf,halfEnaive,MAX_DECODE_ITERATIONS,EXP_NAME))
    
    }
   
    
    
    val decodeTimeMillis = System.currentTimeMillis() - startTime

    //TODO add if debug == true for this test
    if (yi != null) {
    //  print(if (decoded.isInverseOf(yi)) "[IsInv]" else "[NotInv]" + "Decoding took : " + Math.round(decodeTimeMillis ) + "ms")
    }
    
    
counter+=1
lastHash=thisyiHash
    return decoded

  }

  def halfEnergyOf(y:yLabels,thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], graph: xData ):Double = {
    
    var outE =0.0
    for ( xidx <- 0 until graph.size){ //TODO turn this into a for yield. For more idiomatic scala. (But does it matter ? ) 
      val curLab=y.d(xidx)
      val unarTMP = thetaUnary(xidx,curLab)
      val pairTMP = graph.getC(xidx).map { neighIDX => thetaPairwise(curLab,y.d(neighIDX)) }.sum
      outE+=unarTMP+pairTMP
    }
    outE
  }
  
  def unpackWeightVec(weightVec: Vector[Double], xFeatureSize: Int, numClasses: Int = 24, padded: Boolean = false): (DenseMatrix[Double], DenseMatrix[Double]) = {
    // Unary features
    val startIdx = 0
    val endIdx = xFeatureSize * numClasses
    assert(weightVec.length>=endIdx)
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
        pairwiseFeatureVec.toDenseMatrix.reshape(numClasses, numClasses)//TODO does this actually recover properly
      }
     assert((unaryPot.size+pairwisePot.size)==weightVec.length)
    (unaryPot, pairwisePot)
  }

  def predictFn(model: StructSVMModel[xData, yLabels], xi: xData): yLabels = {
    return oracleFn(model, xi, yi = null)
  }
}
