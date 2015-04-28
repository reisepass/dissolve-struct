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


object GraphSegmentation extends DissolveFunctions[GraphStruct[Vector[Double], (Int,Int,Int)],GraphLabels] with Serializable{
  var DISABLE_PAIRWISE: Boolean = true
  type xData = GraphStruct[Vector[Double], (Int,Int,Int)]
  type yLabels = GraphLabels
  
  println("GraphSegmentation::: (DISABLE_PAIRWISE= "+DISABLE_PAIRWISE+" )")
  //Convert the graph into one big feature vector 
  def featureFn(xDat: xData, yDat: yLabels): Vector[Double] = {
    assert(xDat.graphNodes.size == yDat.d.size)
    
    val xFeatures = xDat.getF(0).size
    val numClasses = yDat.numClasses
    
    val unaryFeatureSize = xFeatures * numClasses
    val pairwiseFeatureSize = numClasses * numClasses
    val phi = if (!DISABLE_PAIRWISE) DenseVector.zeros[Double](unaryFeatureSize + pairwiseFeatureSize) else DenseVector.zeros[Double](unaryFeatureSize)

    
    val unary = DenseVector.zeros[Double](unaryFeatureSize)
    for ( idx <- 0 until yDat.d.size){
        val label = yDat.d(idx)
         val startIdx = label * xFeatures
        val endIdx = startIdx + xFeatures
        unary(startIdx until endIdx) := xDat.getF(idx) + unary(startIdx until endIdx)
    }
    
    phi(0 until (unaryFeatureSize)) := unary
       
    if (!DISABLE_PAIRWISE) {
      val pairwise = normalize(getPairwiseFeatureMap(yDat, xDat).toDenseVector) //TODO does this toDenseVector actually use proper columnIndex form, or atleast is it deterministic ? 
      assert((phi.size-unaryFeatureSize) == pairwise.size)
      phi((unaryFeatureSize) until phi.size) := pairwise
    }
    
    phi
  }
  
  // Count pairwise occurances of classes. This is normalized on the outside 
  def getPairwiseFeatureMap(yi : yLabels, xi : xData): DenseMatrix[Double] = {
    
      val pairwiseMat = DenseMatrix.zeros[Double](yi.numClasses, yi.numClasses)
      for ( idx <- 0 until xi.size){
         val myLabel = yi.d(idx)
         xi.getC(idx).foreach { neighbour => {pairwiseMat(myLabel,yi.d(neighbour.idx)) += 1 }}
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
  
  def xFeatureStack (xi:xData) : DenseMatrix[Double] = {
     val featureStack = DenseMatrix.zeros[Double](xi.getF(0).size, xi.size)
     for ( i <- 0 until xi.size){
       featureStack(::,i) := xi.getF(i)
     }
     featureStack
  }

  
 def decodeFn(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], graph : xData, debug: Boolean = false): yLabels = {

    val model = new ItemizedModel
    val numRegions: Int = thetaUnary.rows
    assert(numRegions==graph.size)
    val numClasses: Int = thetaUnary.cols
    
    class RegionVar(val score: Int) extends IntegerVariable(score)

    object PixelDomain extends DiscreteDomain(numClasses)

    class Pixel(i: Int) extends DiscreteVariable(i) { //i is just the initial value 
      def domain = PixelDomain
    }
      
   val labelParams = Array.fill(graph.size){new Pixel(0)}
   val nodePairsUsed: HashSet[(Int,Int)] = new HashSet[(Int,Int)]() 
   for ( idx <- 0 until labelParams.length){
     model ++= new Factor1(labelParams(idx)) {
        def score(k: Pixel#Value) = thetaUnary(idx, k.intValue)  
      } 
     
     if (!DISABLE_PAIRWISE){
       
       graph.getC(idx).foreach { neighbour => {
         if(!nodePairsUsed.contains((idx,neighbour.idx)) && !nodePairsUsed.contains((neighbour.idx,idx))){ //This prevents adding neighbours twice 
         nodePairsUsed.add((idx,neighbour.idx))
         model ++= new Factor2(labelParams(idx), labelParams(neighbour.idx)) {
            def score(i: Pixel#Value, j: Pixel#Value) = thetaPairwise(i.intValue, j.intValue)
         }
         }
       }}
     }
   }
   
   
    
  //  print("nodePairsFound" +nodePairsUsed.size+" Input thetaUnary("+thetaUnary.rows+","+thetaUnary.cols+")/nFactor Graph Size: "+model.factors.size)//TODO remove 
    val maxIterations = if (DISABLE_PAIRWISE) 100 else 1000
    val maximizer = new MaximizeByMPLP(maxIterations)
    val assgn = maximizer.infer(labelParams, model).mapAssignment //Where the magic happens 
     val assgnMean = cc.factorie.infer.InferByMeanField.infer(labelParams,model)
    // Retrieve assigned labels from these pixels

    val out = Array.fill[Int](graph.size)(0)
    for( i <- 0 until out.size){
      out(i) = assgn(labelParams(i)).intValue
    }


    new GraphLabels( Vector(out),numClasses)
  }

 
  def oracleFn(model: StructSVMModel[xData, yLabels], xi: xData, yi: yLabels): yLabels = {
    
    

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
    val decoded = decodeFn(thetaUnary, thetaPairwise, xi, debug = false)
    val decodeTimeMillis = System.currentTimeMillis() - startTime
    
    //TODO add if debug == true for this test
    if ( yi != null) {
    print( if(decoded.isInverseOf(yi)) "[IsInv]" else "[NotInv]" +  "Decoding took : " + Math.round(decodeTimeMillis/1000) +"s")
    }
    
    return decoded
    
  }
  


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
     
    
    def predictFn(model: StructSVMModel[xData, yLabels], xi: xData): yLabels = {
    return oracleFn(model, xi, yi = null)
  }


}