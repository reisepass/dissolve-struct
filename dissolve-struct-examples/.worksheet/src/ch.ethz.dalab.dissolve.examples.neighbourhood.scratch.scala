package ch.ethz.dalab.dissolve.examples.neighbourhood

import ch.ethz.dalab.dissolve.examples.neighbourhood._

object scratch {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(171); 
  println("Welcome to the Scala worksheet");$skip(24); 
  
  println("whaaaat");$skip(41); 
  
  val someD =  GraphUtils.randomVec();System.out.println("""someD  : Array[Double] = """ + $show(someD ));$skip(43); 
  val someD3 = GraphUtils.d3randomVecInt();System.out.println("""someD3  : ch.ethz.dalab.dissolve.examples.neighbourhood.GraphUtils.D3ArrInt = """ + $show(someD3 ));$skip(67); 
  val aHist =  GraphUtils.simple3dhist(someD3,5,Math.round(255/5));System.out.println("""aHist  : breeze.linalg.Vector[Double] = """ + $show(aHist ));$skip(12); val res$0 = 
  aHist sum;System.out.println("""res0: Double = """ + $show(res$0))}
    
  
 // val some = Array.fill(10,10,10){(Math.random()*255).asInstanceOf[Int]}
  
  }
}
