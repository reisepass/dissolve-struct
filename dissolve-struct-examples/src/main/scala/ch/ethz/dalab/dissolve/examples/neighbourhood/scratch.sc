package ch.ethz.dalab.dissolve.examples.neighbourhood

import ch.ethz.dalab.dissolve.examples.neighbourhood._

object scratch {
  println("Welcome to the Scala worksheet")
  
  println("whaaaat")
  
  val someD =  GraphUtils.randomVec()
  val someD3 = GraphUtils.d3randomVecInt()
  val aHist =  GraphUtils.simple3dhist(someD3,5,Math.round(255/5))
  aHist sum
    
  
 // val some = Array.fill(10,10,10){(Math.random()*255).asInstanceOf[Int]}
  
  }
}