package ch.ethz.dalab.dissolve.examples.neighbourhood




import cc.factorie.la._
import scala.collection.mutable.ArrayBuffer
import scala.collection.Set
import cc.factorie.util.{DoubleSeq, RangeIntSeq, SparseDoubleSeq}
import scala.collection.mutable
import cc.factorie.variable._
import cc.factorie.model._
import cc.factorie.maths
import cc.factorie.infer._


/**
 * @author mort
 */
class MaximizeByBPLoopy_rw(numIterations:Int=10) extends MaximizeByBP with Serializable{
    def inferLoopyMax(summary: BPSummary): Unit = {
    for (iter <- 0 to numIterations) { // TODO Make a more clever convergence detection
      for (bpf <- summary.bpFactors) {
        for (e <- bpf.edges) e.bpVariable.updateOutgoing(e)  // get all the incoming messages
        for (e <- bpf.edges) e.bpFactor.updateOutgoing(e)    // send messages
      }
    }
    val assignment = summary.maximizingAssignment
    new MAPSummary(assignment, summary.factors.get.toVector)
  }
    
  def infer(variables:Iterable[DiscreteVar], model:Model, marginalizing:Summary=null) = {
    if (marginalizing ne null) throw new Error("Marginalizing case not yet implemented.")
    val summary = LoopyBPSummaryMaxProduct(variables, BPMaxProductRing, model)
    inferLoopyMax(summary)
    summary
  }
  def apply(varying:Set[DiscreteVar], model:Model): BPSummary = {
    val summary = LoopyBPSummaryMaxProduct(varying, BPMaxProductRing, model)
    inferLoopyMax(summary)
    summary
  }
}

