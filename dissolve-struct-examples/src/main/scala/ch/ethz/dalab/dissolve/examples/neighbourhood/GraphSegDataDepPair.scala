package ch.ethz.dalab.dissolve.examples.neighbourhood

/**
 * @author mort
 */



import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Buffer
import scala.collection.mutable.HashMap
import scala.io.Source
import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.normalize
import breeze.linalg.norm
import breeze.numerics._
import breeze.linalg._
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
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import scala.collection.mutable.HashSet
import cc.factorie.infer.MaximizeByBPLoopy
import cc.factorie.la.DenseTensor1
import cc.factorie.la.Tensor


//This class assumes that at index 0 of the xFeature vector is the average intensity of the superpixel 
class GraphSegDataDepPair(dataDepBinFn:((Node[Vector[Double]],Node[Vector[Double]])=>Int),dataDepNumBins:Int,EXP_NAME:String="NoName", classFreqs:Map[Int,Double]=null, LOSS_AUGMENTATION_OVERRIDE: Boolean=false, PAIRWISE_UPPER_TRI:Boolean=true,  loopyBPmaxIter:Int=10, alsoWeighLossAugByFreq:Boolean=false) extends DissolveFunctions[GraphStruct[Vector[Double], (Int, Int, Int)], GraphLabels] with Serializable {
    
  type xData = GraphStruct[Vector[Double], (Int, Int, Int)]
  type yLabels = GraphLabels
  
  val myLoopyBP = new MaximizeByBPLoopy_rw(loopyBPmaxIter)
  
   
  val labelIDs = if(classFreqs!=null)  classFreqs.keySet.toList.sorted else List(0,1)
  if(classFreqs!=null)  assert(labelIDs==(0 until labelIDs.length).toList)
  val lableFreqLoss = if(classFreqs!=null)  DenseVector(labelIDs.map { labl => classFreqs.get(labl).get }.toArray) else DenseVector(Array.fill(1){0.0})
  
  
  
  //Convert the graph into one big feature vector 
  def featureFn(xDat: xData, yDat: yLabels): Vector[Double] = {
    assert(xDat.graphNodes.size == yDat.d.size)
    

    val xFeatures = xDat.getF(0).size 
    val numClasses = yDat.numClasses

     
    val unaryFeatureSize = (xFeatures * numClasses)
    val pairwiseFeatureSize = numClasses * numClasses *dataDepNumBins
    val phi =  DenseVector.zeros[Double](unaryFeatureSize + pairwiseFeatureSize) 

    
    
    val unary = DenseVector.zeros[Double](unaryFeatureSize)
    for (idx <- 0 until yDat.d.size) {
      val label = yDat.d(idx)
      val startIdx = label * xFeatures
      val endIdx = startIdx + xFeatures
      val curF =xDat.getF(idx) 
      unary(startIdx until endIdx) :=curF + unary(startIdx until endIdx)
    }

    
      phi(0 until (unaryFeatureSize)) := unary
    

    
      val pairwise = getPairwiseFeatureMap(yDat, xDat)
      
      
      phi((unaryFeatureSize) until phi.size) := pairwise
   phi
  }
  
  
  def getPairwiseFeatureMap(yi: yLabels, xi: xData): DenseVector[Double] = {

    val pairwiseMats = Array.fill(dataDepNumBins){DenseMatrix.zeros[Double](yi.numClasses, yi.numClasses)}
    for (idx <- 0 until xi.size) {
      val myLabel = yi.d(idx)
      xi.getC(idx).foreach { neighbour => { 
        val dataDepBin=dataDepBinFn(xi.get(idx),xi.get(neighbour))
        pairwiseMats(dataDepBin)(myLabel, yi.d(neighbour)) += 1; pairwiseMats(dataDepBin)(yi.d(neighbour),myLabel) += 1 
        
      }}
    }
    
    
      
  val matS = yi.numClasses*yi.numClasses
  val outAll = DenseVector.zeros[Double](yi.numClasses*yi.numClasses*dataDepNumBins)
  for ( i <- 0 until dataDepNumBins) yield{
    val startIdx = i*matS
    val endIdx = startIdx + matS
    outAll(startIdx until endIdx):= normalize(pairwiseMats(i).toDenseVector)
  }
  
  val alldem = for ( i <- 1 until dataDepNumBins) yield{
    normalize(pairwiseMats(i).toDenseVector)
  }
  val concat = alldem.foldLeft(normalize(pairwiseMats(0).toDenseVector))( (r,c) => DenseVector.vertcat(r,c) )
  assert(concat.equals(outAll),"FAIL \n"+outAll.toString()+" \n "+concat.toString) //TODO REMOVE DEBUG just checking if this is how fold works
  concat
  }

  def lossFn(yTruth: yLabels, yPredict: yLabels): Double = {
    assert(yPredict.d.size == yTruth.d.size)
    //val classFreqs = yTruth.classFreq

    val loss =
      for (
        idx <- 0 until yPredict.d.size
      ) yield {
        //TODO put this back to weighted loss 
        val classLossWeight = if(classFreqs==null) 1 else classFreqs.get(yTruth.d(idx)).get
        // if (yTruth.get(x, y, z) == yPredict.get(x, y, z)) 0.0 else 1.0 / classFreqs.get(yTruth.get(x, y, z)).get // Insert classFrequency back into the truthObject
        if (yTruth.d(idx) == yPredict.d(idx)) 0.0 else 1.0/classLossWeight // Insert classFrequency back into the truthObject
      }

    loss.sum / (yPredict.d.size)
  }
  
  
  /*
  def lossFn(yTruth: yLabels, yPredict: yLabels): Double = {
    0.0
  }
  * 
  */

  def xFeatureStack(xi: xData): DenseMatrix[Double] = {
    val featureStack = DenseMatrix.zeros[Double](xi.getF(0).size, xi.size)
    for (i <- 0 until xi.size) {
      featureStack(::, i) := xi.getF(i)
    }
    featureStack
  }

  def decodeFn_BP(thetaUnary: DenseMatrix[Double], thetaDepPairwise: Array[DenseMatrix[Double]], graph: xData, conGraid:Array[Map[Int,Int]], debug: Boolean = false): yLabels = {

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
          val weights: DenseTensor1 = new DenseTensor1(thetaUnary(idx, ::).t.toArray)
        def score(k: Pixel#Value) = thetaUnary(idx, k.intValue) //TODO am i reading thetaUnary corectly here? Maybe its witched
        override def valuesScore(tensor: Tensor): Double = {
          weights dot tensor
        }
      }

      
        
        def getPair (i:Int,j:Int,bin:Int):Double={
            val left = min(i,j)
            val right = max(i,j)
            thetaDepPairwise(bin)(left,right)
        }

        graph.getC(idx).foreach { neighbour =>
          {
            if (!nodePairsUsed.contains((idx, neighbour)) && !nodePairsUsed.contains((neighbour, idx))) { //This prevents adding neighbours twice 
              nodePairsUsed.add((idx, neighbour))
              model ++= new Factor2(labelParams(idx), labelParams(neighbour)) {
                val gBin = conGraid(idx).get(neighbour).get
                val weights: DenseTensor1 = new DenseTensor1(thetaDepPairwise(gBin).toArray)
                def score(i: Pixel#Value, j: Pixel#Value) = getPair(i.intValue, j.intValue,gBin)
                  override def valuesScore(tensor: Tensor): Double = {
                    weights dot tensor
                  }
              }
            }
          }
        }
      
    }

   
    if(counter<1) println("nodePairsFound" +nodePairsUsed.size+" Input thetaUnary("+thetaUnary.rows+","+thetaUnary.cols+")/nFactor Graph Size: "+model.factors.size)//TODO remove 
     
    myLoopyBP.maximize(labelParams,model)
    val mapLabelsBP: Array[Int] = (0 until numRegions).map {
      idx =>
        // assgn(pixelSeq(idx)).intValue
        labelParams(idx).intValue
    }.toArray

    
   
    val t1 = System.currentTimeMillis()
    //print(" decodeTime=[%d s]".format(  (t1-t0)/1000  ))
    new GraphLabels(Vector(mapLabelsBP), numClasses)
  }

  
  var counter =0; //TODO REMOVE 
  var lastHash:Int=0;
  
  def computeConnectionGradient(xi:xData):Array[Map[Int,Int]]={
    
    
    val out=for(i <- 0 until xi.size) yield{
      
     val mutHashMap = new HashMap[Int,Int]
     xi.getC(i).foreach { neigh => { mutHashMap.put(neigh,dataDepBinFn(xi.get(i),xi.get(neigh)))} }
     mutHashMap.toMap
    }
    out.toArray
  }
  
  def oracleFn(model: StructSVMModel[xData, yLabels], xi: xData, yi: yLabels): yLabels = {
    
    
    
    
    
    
    val thisyiHash= if(yi!=null) yi.hashCode else 0
    val numClasses = model.numClasses

    val numDims = xi.getF(0).length

    val weightVec = model.getWeights()

    // Unary is of size f x K, each column representing feature vector of class K
    // Pairwise if of size K * K
    val (unaryWeights, pairwiseWeights) = unpackWeightVec(weightVec, numDims, numClasses = numClasses)
    assert(unaryWeights.rows == numDims)
    assert(unaryWeights.cols == numClasses)
    assert(pairwiseWeights(0).rows == pairwiseWeights(0).cols)
    assert(pairwiseWeights(0).rows == numClasses)
    
   

    val phi_Y: DenseMatrix[Double] = xFeatureStack(xi) // Retrieves a f x r matrix representation of the original image, i.e, each column is the feature vector that region r
    val thetaUnary = phi_Y.t * unaryWeights // Returns a r x K matrix, where theta(r, k) is the unary potential of region r having label k
    //Aureliens code calls this Unary potential (without the loss, but its ok) 
    
    //TODO check if this transpose makes sense 

    val thetaPairwise = pairwiseWeights
    val connectionGradient = computeConnectionGradient(xi)

    
    // If yi is present, do loss-augmentation
    if (yi != null && !LOSS_AUGMENTATION_OVERRIDE) {
      val freqLoss = if(alsoWeighLossAugByFreq) (lableFreqLoss:*=(1.0 / xi.size)) else (DenseVector(Array.fill(numClasses){1.0 / xi.size}))
      for (idx <- 0 until xi.size) { //TODO check if this is using correct indexs 
        
        thetaUnary(idx, ::) := thetaUnary(idx, ::) + freqLoss.t  //We are using a zero-one loss per y so here there are just constants
        // Loss augmentation step
        val k = yi.d(idx)
        thetaUnary(idx, k) = thetaUnary(idx, k) - ( freqLoss(k)) //This is a zero loss b/c it cancels out the +1 for all non correct labels 
        //This zero one loss is repeated code from the lossFn. lossFn gets loss for  
        //     a whole image but inside it uses zero-one loss for pixel comparison 
        //     this should be a function so it is common between the two uses 

      }

    }

    
    
    /**
     * Parameter estimation
     */
    val startTime = System.currentTimeMillis()
   
    
    
  
   
      val t00 = System.currentTimeMillis()
     // val factD = decodeFn_sample(thetaUnary, thetaPairwise, xi, debug = false)
      val t0BP = System.currentTimeMillis()
      val decoded = decodeFn_BP(thetaUnary, thetaPairwise, xi,connectionGradient)
     // println("#CMP Factorie BP< Ft(" +(t0BP-t00)+") BPt("+(System.currentTimeMillis()-t0BP)+") dif: " + lossFn(bpD, factD) +" >")
      
      
    
    
    
    
    val decodeTimeMillis = System.currentTimeMillis() - startTime

  
    
counter+=1
lastHash=thisyiHash
    return decoded
  }

  
  def unpackWeightVec(weightVec: Vector[Double], xFeatureSize: Int, numClasses:Int): (DenseMatrix[Double], Array[DenseMatrix[Double]]) = {
    // Unary features
    val startIdx = 0
    val endIdx = xFeatureSize * numClasses
    //assert(weightVec.length>=endIdx)
    val unaryFeatureVec = weightVec(startIdx until endIdx).toDenseVector // Stored as [|--f(k=0)--||--f(k=1)--| ... |--f(K=k)--|]
    val unaryPot = unaryFeatureVec.toDenseMatrix.reshape(xFeatureSize, numClasses)

    // Pairwise feature Vector
    val pairwiseSize=numClasses*numClasses
    val pairwiseMats = Array.fill(dataDepNumBins){DenseMatrix.zeros[Double](numClasses,numClasses)}
 
        val unaryEnd = endIdx
        for(i <- 0 until dataDepNumBins){
          val startI = unaryEnd + ( i*pairwiseSize)
          val endI = startI + pairwiseSize
          val pairwiseFeatureVec = weightVec(startI until endI).toDenseVector
           assert(pairwiseFeatureVec.size == numClasses * numClasses, "was ="+pairwiseFeatureVec.size  +" should have been= "+(numClasses * numClasses))
           pairwiseMats(i)=pairwiseFeatureVec.toDenseMatrix.reshape(numClasses, numClasses)
        }
        
    (unaryPot, pairwiseMats)
  }

  def predictFn(model: StructSVMModel[xData, yLabels], xi: xData): yLabels = {
    return oracleFn(model, xi, yi = null)
  }
}
