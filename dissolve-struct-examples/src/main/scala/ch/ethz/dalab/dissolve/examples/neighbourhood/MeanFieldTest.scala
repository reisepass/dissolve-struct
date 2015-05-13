package ch.ethz.dalab.dissolve.examples.neighbourhood

import breeze.linalg._
import breeze.stats.DescriptiveStats._
import breeze.stats._

object MeanFieldTest {
  type xData = GraphStruct[Vector[Double], (Int, Int, Int)]
  

  //Our Q distribution is a discrete PDF over all classes per node. 
  //Need a Working memory of current probability of a given class for each Node in the graph. 
  def decodeFn(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], graph: xData,learningRate:Double = 0.1, maxIterations:Int=100, debug: Boolean = false): GraphLabels = {
    val thetaMin = min(thetaPairwise.toArray)
    val thetaMax = max(thetaPairwise)
    val probTheta = (thetaPairwise-thetaMin)/(thetaMax-thetaMin) //TODO conforming theta to be a probability like this is pretty bad. I dont think we can solve a non probabiltiy with mean field 
    
    val t0 = System.currentTimeMillis()

    val numRegions: Int = thetaUnary.rows
    assert(numRegions == graph.size)
    val numClasses: Int = thetaUnary.cols
    
    assert(learningRate>0&&learningRate<=1)
    val Q = DenseMatrix.ones[Double](numRegions, numClasses) //TODO maybe initialize to 1/num neighbours or 1/numneighbours*numClass
    //TODO This Q could be stored in a spark RDD if it gets too large for memory;; Prob not necesary because thetaUnayr is the same size 
    //TODO consider storing the probabilities in a byte
    

    
    for (iter <- 0 until maxIterations) {
      for (xi <- 0 until graph.size) {
        val tI =  System.currentTimeMillis()
        for( xiLab <- 0 until numClasses){
        val neigh = graph.getC(xi).toArray
        val neighIdx = neigh.map { x => x }
        val allNeighClass = (0 until numClasses).toList.combinations(neighIdx.length)
        val p_d_given_xi = thetaUnary(xi,xiLab) 
        
        val newQest = allNeighClass.foldLeft(0.0){  (runingSum,curLabels)=>
          {
            val labelAndIdx = curLabels zip neighIdx
            val qProd = labelAndIdx.map{ case (aLabel,qID)=>Q(qID,aLabel) }.product
            val p_x_prior = labelAndIdx.map{ case (nLabel,nID)=> probTheta(xiLab,nLabel) }.product
            //TODO error, p_x_prior can be negative do to its multiplicaiton with W, hence we need to normalize this 
            runingSum+qProd*Math.log(p_d_given_xi*p_x_prior)
          }
        }
        if(iter==4){
          assert(true)
        }
        Q(xi,xiLab)=Q(xi,xiLab)*(1-learningRate)+(learningRate)*Math.exp(newQest) //TODO make the learning rate decrease as iter increases         
        }
        val tII =  System.currentTimeMillis()
       //print("<X=[%d ms] >".format(  (tII-tI)  ))

      }
    }
    val outLab = Vector(Array.fill(numRegions)(0))
    for (row <- 0 until Q.rows) {
      outLab(row)=argmax(Q(row,::))
    }
   
    val t1 = System.currentTimeMillis()
    print("[MF decodeTime=[%d s] ]".format(  (t1-t0)/1000  ))
    return ( new GraphLabels(outLab,numClasses))
  }
  
  //Our Q distribution is a discrete PDF over all classes per node. 
  //Need a Working memory of current probability of a given class for each Node in the graph. 
  def decodeFn_AL(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], graph: xData,learningRate:Double = 0.1, maxIterations:Int=100, debug: Boolean = false): GraphLabels = {
    val thetaMin = min(thetaPairwise.toArray)
    val thetaMax = max(thetaPairwise)
   // val probTheta = (thetaPairwise-thetaMin)/(thetaMax-thetaMin) //TODO conforming theta to be a probability like this is pretty bad. I dont think we can solve a non probabiltiy with mean field 
    
    val t0 = System.currentTimeMillis()

    val numRegions: Int = thetaUnary.rows
    assert(numRegions == graph.size)
    val numClasses: Int = thetaUnary.cols
    
    assert(learningRate>0&&learningRate<=1)
    val Q = DenseMatrix.ones[Double](numRegions, numClasses) //TODO maybe initialize to 1/num neighbours or 1/numneighbours*numClass
    //TODO This Q could be stored in a spark RDD if it gets too large for memory;; Prob not necesary because thetaUnayr is the same size 
    //TODO consider storing the probabilities in a byte
    

    
    for (iter <- 0 until maxIterations) {
      for (xi <- 0 until graph.size) {
        val tI =  System.currentTimeMillis()
        for( xiLab <- 0 until numClasses){
          
          val neigh = graph.getC(xi).toArray
          val allClasses = (0 until numClasses).toList
          val newQest = neigh.toList.map { neighIdx =>
            allClasses.foldLeft(0.0) { (running, curClass) =>
              {
                running + Q(neighIdx, curClass) * Math.exp(thetaPairwise(curClass, xiLab)) * Math.exp(thetaUnary(xi, xiLab))
              }
            }
          }.sum

          Q(xi,xiLab)=Q(xi,xiLab)*(1-learningRate)+(learningRate)*(newQest) //TODO make the learning rate decrease as iter increases         
        }
        val tII =  System.currentTimeMillis()
       //print("<X=[%d ms] >".format(  (tII-tI)  ))

      }
    }
    val outLab = Vector(Array.fill(numRegions)(0))
    for (row <- 0 until Q.rows) {
      outLab(row)=argmax(Q(row,::))
    }
   
    val t1 = System.currentTimeMillis()
    print("[MF decodeTime=[%d s] ]".format(  (t1-t0)/1000  ))
    return ( new GraphLabels(outLab,numClasses))
  }
   def decodeFn_AR(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], graph: xData, maxIterations:Int=100, debug: Boolean = false, temp:Double=5): GraphLabels = {
   val DISABLE_PAIRWISE = if(thetaPairwise.size==0) true else false 
   // val probTheta = (thetaPairwise-thetaMin)/(thetaMax-thetaMin) //TODO conforming theta to be a probability like this is pretty bad. I dont think we can solve a non probabiltiy with mean field 
    
    val t0 = System.currentTimeMillis()

    val numRegions: Int = thetaUnary.rows
    assert(numRegions == graph.size)
    val numClasses: Int = thetaUnary.cols
    
    
    val Q = DenseMatrix.ones[Double](numRegions, numClasses) //TODO maybe initialize to 1/num neighbours or 1/numneighbours*numClass
    //TODO This Q could be stored in a spark RDD if it gets too large for memory;; Prob not necesary because thetaUnayr is the same size 
    //TODO consider storing the probabilities in a byte
    

    
    for (iter <- 0 until maxIterations) {
      for (xi <- 0 until graph.size) {
        val tI =  System.currentTimeMillis()
        for( xiLab <- 0 until numClasses){
          
          val neigh = graph.getC(xi).toArray
          val allClasses = (0 until numClasses).toList
          
          val newQest = neigh.toList.map { neighIdx =>
            allClasses.foldLeft(0.0) { (running, curClass) =>
              {
                running +  Math.exp(Q(neighIdx, curClass)*(if(DISABLE_PAIRWISE)0 else thetaPairwise(curClass, xiLab))) * Math.exp((1/temp)*thetaUnary(xi, xiLab))
              }
            }
          }.sum

          Q(xi,xiLab)=(1/temp)*newQest     
        }
        val tII =  System.currentTimeMillis()
       //print("<X=[%d ms] >".format(  (tII-tI)  ))

      }
    }
    val outLab = Vector(Array.fill(numRegions)(0))
    for (row <- 0 until Q.rows) {
      outLab(row)=argmax(Q(row,::))
    }
   
    val t1 = System.currentTimeMillis()
   // print("[MF decodeTime=[%d s] ]".format(  (t1-t0)/1000  ))
    return ( new GraphLabels(outLab,numClasses))
  }



}