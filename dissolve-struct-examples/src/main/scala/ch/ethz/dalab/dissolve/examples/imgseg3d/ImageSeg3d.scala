package ch.ethz.dalab.dissolve.examples.imgseg3d

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

object ImageSeg3d {

  /*
   * Counts occurances of adjacent pairs of classes 
   * 
   * 
   */

  def getPairwiseFeatureMap(yMat: ThreeDimMat[Int], xMat: ThreeDimMat[Array[Double]]): DenseMatrix[Double] = {

    assert(xMat.xDim == yMat.xDim)
    assert(xMat.yDim == yMat.yDim)
    assert(xMat.zDim == yMat.zDim)

    val numFeatures = xMat.get(0, 0, 0).length //TODO this hist feature size uniformity is not garateed inside my datastructure 
    val numClasses = yMat.classSet.size
    val numRegions = xMat.xDim * xMat.yDim * xMat.zDim

    val pairwiseMat = DenseMatrix.zeros[Double](numClasses, numClasses)

    for (
      y <- 0 until xMat.xDim;
      x <- 0 until xMat.yDim;
      z <- 0 until xMat.zDim
    ) {
      val classA = yMat.get(x, y, z)

      val neighbours = List((1, 0, 0), (0, 1, 0), (0, 0, 1))

      for ((dx, dy, dz) <- neighbours if (x + dx >= 0) && (y + dy >= 0) && (z + dz >= 0) && (x + dx < xMat.xDim) && (y + dy < xMat.yDim) && (z + dz < xMat.zDim)) {
        val classB = yMat.get(x + dx, y + dy, z + dz)
        pairwiseMat(classA, classB) += 1.0
        pairwiseMat(classB, classA) += 1.0
      }
    }

    pairwiseMat
  }

  /**
   * Loss function
   */
  def lossFn(yTruth: Array[Array[Array[Int]]], yPredict: Array[Array[Array[Int]]]): Double = {

    val xDimTr = yTruth.length;
    val yDimTr = yTruth(0).length;
    val zDimTr = yTruth(0)(0).length;
    val xDimPr = yTruth.length;
    val yDimPr = yTruth(0).length;
    val zDimPr = yTruth(0)(0).length;
    assert(xDimPr == xDimTr)
    assert(yDimPr == yDimTr)
    assert(zDimPr == zDimTr)

    val loss =
      for (
        y <- 0 until xDimTr;
        x <- 0 until yDimTr;
        z <- 0 until zDimTr
      ) yield {
        if (yTruth(x)(y)(z) == yPredict(x)(y)(z)) 0.0 else 1.0 // yTruth(x, y).classFrequency  Insert classFrequency back into the truthObject
      }

    loss.sum / (xDimTr * yDimTr * zDimTr)
  }

  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")

    val options: Map[String, String] = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt)    => (opt -> "true")
        case _             => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    System.setProperty("spark.akka.frameSize", "512")
    println(options)

  }
}


