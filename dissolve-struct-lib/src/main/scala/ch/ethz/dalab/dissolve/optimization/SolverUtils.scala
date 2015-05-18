package ch.ethz.dalab.dissolve.optimization

import ch.ethz.dalab.dissolve.classification.StructSVMModel
import breeze.linalg._
import ch.ethz.dalab.dissolve.regression.LabeledObject
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import scala.reflect.ClassTag

object SolverUtils {

  /**
   * Average loss
   */
  def averageLoss[X, Y](data: Seq[LabeledObject[X, Y]],
                        dissolveFunctions: DissolveFunctions[X, Y],
                        model: StructSVMModel[X, Y]): Double = {

    var lossTerm: Double = 0.0
    for (i <- 0 until data.size) {
      val ystar_i = dissolveFunctions.predictFn(model, data(i).pattern)
      lossTerm += dissolveFunctions.lossFn(data(i).label, ystar_i)
    }

    // Return average of loss terms
    lossTerm / (data.size.toDouble)
  }

  def averageLoss[X, Y](data: RDD[LabeledObject[X, Y]],
                        dissolveFunctions: DissolveFunctions[X, Y],
                        model: StructSVMModel[X, Y],
                        dataSize: Int): Double = {

    val loss =
      data.map {
        case datapoint =>
          val ystar_i = dissolveFunctions.predictFn(model, datapoint.pattern)
          dissolveFunctions.lossFn(datapoint.label, ystar_i)
      }.fold(0.0)((acc, ele) => acc + ele)

    loss / dataSize
  }

  /**
   * Objective function (SVM dual, assuming we know the vector b of all losses. See BCFW paper)
   */
  def objectiveFunction(w: Vector[Double],
                        b_alpha: Double,
                        lambda: Double): Double = {
    // Return the value of f(alpha)
    0.5 * lambda * (w.t * w) - b_alpha
  }

  /**
   * Compute Duality gap
   * Requires one full pass of decoding over all data examples.
   */
  def dualityGap[X, Y](data: Seq[LabeledObject[X, Y]],
                       featureFn: (X, Y) => Vector[Double],
                       lossFn: (Y, Y) => Double,
                       oracleFn: (StructSVMModel[X, Y], X, Y) => Y,
                       model: StructSVMModel[X, Y],
                       lambda: Double)(implicit m: ClassTag[Y]): (Double, Vector[Double], Double) = {

    val phi = featureFn
    val maxOracle = oracleFn

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    val n: Int = data.size
    val d: Int = model.getWeights().size
    val yStars = new Array[Y](n)

    for (i <- 0 until n) {
      yStars(i) = maxOracle(model, data(i).pattern, data(i).label)
    }

    var w_s: DenseVector[Double] = DenseVector.zeros[Double](d)
    var ell_s: Double = 0.0
    for (i <- 0 until n) {
      w_s += phi(data(i).pattern, data(i).label) - phi(data(i).pattern, yStars(i))
      ell_s += lossFn(data(i).label, yStars(i))
    }

    w_s = w_s / (lambda * n)
    ell_s = ell_s / n

    val gap: Double = w.t * (w - w_s) * lambda - ell + ell_s

    (gap, w_s, ell_s)
  }

  /**
   * Alternative implementation, using fold. TODO: delete this or the above
   * Requires one full pass of decoding over all data examples.
   */
  def dualityGap[X, Y](data: RDD[LabeledObject[X, Y]],
                       dissolveFunctions: DissolveFunctions[X, Y],
                       model: StructSVMModel[X, Y],
                       lambda: Double,
                       dataSize: Int)(implicit m: ClassTag[Y]): (Double, Vector[Double], Double) = {

    val phi = dissolveFunctions.featureFn _
    val maxOracle = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    val n: Int = dataSize.toInt
    val d: Int = model.getWeights().size

    val (w_s_orig, ell_s_orig) = data.map {
      case datapoint =>
        val yStar = maxOracle(model, datapoint.pattern, datapoint.label)
        val w_s = phi(datapoint.pattern, datapoint.label) - phi(datapoint.pattern, yStar)
       
       
        val ell_s = lossFn(datapoint.label, yStar)
        if(ell_s >1)
          println("loss greater than 1 ??")
          
        (w_s, ell_s)
    }.fold((Vector.zeros[Double](d), 0.0)) {
      case ((w_acc, ell_acc), (w_i, ell_i)) =>
        (w_acc + w_i, ell_acc + ell_i)
    }
   // val oldell_s=ell_s
    val w_s = w_s_orig / (lambda * n)
    val ell_s = ell_s_orig / n

    val gap: Double = w.t * (w - w_s) * lambda - ell + ell_s
    if(gap<0){
      val size = data.map{case datapoint => 1}.fold(0){case(last, a)=>(last+a)}
      
      println("gap can never be negative")
    }
    (gap, w_s, ell_s)
  }

  /**
   * Primal objective.
   * Requires one full pass of decoding over all data examples.
   */
  def primalObjective[X, Y](data: Vector[LabeledObject[X, Y]],
                            dissolveFunctions: DissolveFunctions[X, Y],
                            model: StructSVMModel[X, Y],
                            lambda: Double): Double = {

    val featureFn = dissolveFunctions.featureFn _
    val oracleFn = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _

    var hingeLosses: Double = 0.0
    for (i <- 0 until data.size) {
      val yStar_i = oracleFn(model, data(i).pattern, data(i).label)
      val loss_i = lossFn(data(i).label, yStar_i)
      val psi_i = featureFn(data(i).pattern, data(i).label) - featureFn(data(i).pattern, yStar_i)

      val hingeloss_i = loss_i - model.getWeights().t * psi_i
      // println("loss_i = %f, other_loss = %f".format(loss_i, model.getWeights().t * psi_i))
      // assert(hingeloss_i >= 0.0)

      hingeLosses += hingeloss_i
    }

    // Compute the primal and return it
    0.5 * lambda * (model.getWeights.t * model.getWeights) + hingeLosses / data.size

  }

  /**
   * Get Spark's properties
   */
  def getSparkConfString(sc: SparkConf): String = {
    val keys = List("spark.app.name", "spark.executor.memory", "spark.task.cpus", "spark.local.dir", "spark.default.parallelism")
    val sb: StringBuilder = new StringBuilder()

    for (key <- keys)
      sb ++= "# %s=%s\n".format(key, sc.get(key, "NA"))

    sb.toString()
  }

}