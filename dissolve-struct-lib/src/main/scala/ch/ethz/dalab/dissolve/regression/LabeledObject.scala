package ch.ethz.dalab.dissolve.regression

import org.apache.spark.mllib.regression.LabeledPoint

import breeze.linalg._

case class LabeledObject[X, Y](
  val label: Y,
  val pattern: X) extends Serializable {
}