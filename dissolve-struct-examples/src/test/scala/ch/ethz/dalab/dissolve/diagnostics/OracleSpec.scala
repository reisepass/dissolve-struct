package ch.ethz.dalab.dissolve.diagnostics

import org.scalatest.FlatSpec
import breeze.linalg._
import scala.collection.mutable.ArrayBuffer
import ch.ethz.dalab.dissolve.regression.LabeledObject

/**
 * @author torekond
 */
class OracleSpec extends UnitSpec {

  val NUM_WEIGHT_VECS = 30 // = # times each test case is attempted
  val EPSILON=(-0.00000001)
  // A sample datapoint
  val lo = data(0)
  // Size of joint feature map
  val d = phi(lo.pattern, lo.label).size
  // No. of data examples
  val M = data.length

  /**
   * Initialize a bunch of weight vectors
   */
  type WeightVector = Vector[Double]
  val weightVectors: Array[WeightVector] = {
    val seq = new ArrayBuffer[WeightVector]

    // All 1's
    seq += DenseVector.ones[Double](d)

    // All 0's
    seq += DenseVector.zeros[Double](d)

    // Few random vectors
    for (k <- 0 until NUM_WEIGHT_VECS - seq.size)
      seq += DenseVector.rand(d)

    seq.toArray
  }

  /**
   * Tests - Structured Hinge Loss
   */

  "Structured Hinge Loss [Δ(y_m, y*) - < w, ψ(x_m, y*) >]" should "be >= 0 for GIVEN (x_m, y_m) pairs" in {

    val shlSeq: Seq[Double] = for (k <- 0 until NUM_WEIGHT_VECS) yield {

      // Set weight vector
      val w: WeightVector = weightVectors(k)
      model.updateWeights(w)

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val y_m = lo.label

      // Get loss-augmented argmax prediction
      val ystar = maxoracle(model, x_m, y_m)
      val realLoss = delta(y_m, ystar)
      val modelLoss = deltaF(lo, ystar, w)
      
      
      val shl = realLoss - modelLoss
          if(shl < EPSILON)  
      println(" shl= "+shl+" rL:" + realLoss + " - mL:" + modelLoss+" ysInv:"+y_m.isInverseOf(ystar)+ " norm(w):"+norm(w.toDenseVector)+" nrom(yStar):"+norm(DenseVector(ystar.d.toArray)))
      shl

    }

    // This should be empty
    val negShlSeq: Seq[Double] = shlSeq.filter(_ < EPSILON)

    assert(negShlSeq.length == 0,
      "%d / %d cases failed".format(negShlSeq.length, shlSeq.length))

  }

  it should "be >= 0 for PERTURBED (x_m, y_m) pairs" in {

    val shlSeq: Seq[Double] = for (k <- 0 until NUM_WEIGHT_VECS) yield {

      // Set weight vector
      val w: WeightVector = weightVectors(k)
      model.updateWeights(w)

      // Sample a random (x, y) pair
      val m = scala.util.Random.nextInt(M)
      val x_m = data(m).pattern
      val y_m = perturb(data(m).label, 0.1) // Perturbed copy
      val lo = LabeledObject(y_m, x_m)

      // Get loss-augmented argmax prediction
      val ystar = maxoracle(model, x_m, y_m)
      val shl = delta(y_m, ystar) - deltaF(lo, ystar, w)

      shl

    }

    // This should be empty
    val negShlSeq: Seq[Double] = shlSeq.filter(_ < EPSILON)

    assert(negShlSeq.length == 0,
      "%d / %d cases failed".format(negShlSeq.length, shlSeq.length))

  }

  /**
   * Tests - Discriminant function
   */
  "F(x_m, y*)" should "be >= F(x_m, y_m)" in {

    val diffSeq = for (k <- 0 until NUM_WEIGHT_VECS) yield {
      // Set weight vector
      val w: WeightVector = weightVectors(k)
      model.updateWeights(w)

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val y_m = lo.label

      // Get loss-augmented argmax prediction
      val ystar = maxoracle(model, x_m, y_m)

      val F_ystar = F(x_m, ystar, w)
      val F_gt = F(x_m, y_m, w)
      if ((F_ystar - F_gt < 0.0)) {
        val ystarIsEq = y_m.d.equals(ystar.d)
        val ysAreInv = y_m.isInverseOf(ystar)
        println(" F_ystar=" + F_ystar + " F_gt=" + F_gt + " y's are equal=" + ystarIsEq + " y's are Inv=" + ysAreInv+" norm(w)="+norm(DenseVector(w.toArray)))

      }
      F_ystar - F_gt
    }

    // This should be empty
    val negDiffSeq: Seq[Double] = diffSeq.filter(_ < EPSILON)

    assert(negDiffSeq.length == 0,
      "%d / %d cases failed".format(negDiffSeq.length, diffSeq.length))

  }

  it should "be >= PERTURBED F(x_m, y_m)" in {

    val diffSeq = for (k <- 0 until NUM_WEIGHT_VECS) yield {
      // Set weight vector
      val w: WeightVector = weightVectors(k)
      model.updateWeights(w)

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val y_m = perturb(lo.label, 0.1)

      // Get loss-augmented argmax prediction
      val ystar = maxoracle(model, x_m, y_m)

      val F_ystar = F(x_m, ystar, w)
      val F_gt = F(x_m, y_m, w)
      if ((F_ystar - F_gt < 0.0)) {
        val ystarIsEq = y_m.d.equals(ystar.d)
        val ysAreInv = y_m.isInverseOf(ystar)
      println(" F_ystar=" + F_ystar + " F_gt=" + F_gt + " y's are equal=" + ystarIsEq + " y's are Inv=" + ysAreInv+" norm(w)="+norm(DenseVector(w.toArray)))

      }

      F_ystar - F_gt
    }

    // This should be empty
    val negDiffSeq: Seq[Double] = diffSeq.filter(_ < EPSILON)

    assert(negDiffSeq.length == 0,
      "%d / %d cases failed".format(negDiffSeq.length, diffSeq.length))

  }

  "H(w; x_m, y_m)" should "be >= Δ(y_m, y_m) + F(x_m, y_m)" in {

    val diffSeq = for (k <- 0 until NUM_WEIGHT_VECS) yield {
      // Set weight vector
      val w: WeightVector = weightVectors(k)
      model.updateWeights(w)

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val y_m = lo.label

      // Get loss-augmented argmax prediction
      val ystar = maxoracle(model, x_m, y_m)

      val H = delta(y_m, ystar) - deltaF(lo, ystar, w)
      val F_loss_aug = delta(y_m, y_m) - deltaF(lo, y_m, w)

      H - F_loss_aug
    }

    // This should be empty
    val negDiffSeq: Seq[Double] = diffSeq.filter(_ < EPSILON)

    assert(negDiffSeq.length == 0,
      "%d / %d cases failed".format(negDiffSeq.length, diffSeq.length))
  }

  it should "be >= PERTURBED Δ(y_m, y_m) + F(x_m, y_m)" in {

    val diffSeq = for (k <- 0 until NUM_WEIGHT_VECS) yield {
      // Set weight vector
      val w: WeightVector = weightVectors(k)
      model.updateWeights(w)

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val y_m = perturb(lo.label, 0.1)

      // Get loss-augmented argmax prediction
      val ystar = maxoracle(model, x_m, y_m)

      val H = delta(y_m, ystar) - deltaF(lo, ystar, w)
      val F_loss_aug = delta(y_m, y_m) - deltaF(lo, y_m, w)

      H - F_loss_aug
    }

    // This should be empty
    val negDiffSeq: Seq[Double] = diffSeq.filter(_ < EPSILON)

    assert(negDiffSeq.length == 0,
      "%d / %d cases failed".format(negDiffSeq.length, diffSeq.length))
  }

}