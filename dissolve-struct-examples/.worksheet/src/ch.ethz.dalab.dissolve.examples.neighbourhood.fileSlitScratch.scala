package ch.ethz.dalab.dissolve.examples.neighbourhood

import scala.sys.process._


object fileSlitScratch {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(152); 
  println("Welcome to the Scala worksheet");$skip(14); val res$0 = 
  
  "pwd".!!;System.out.println("""res0: String = """ + $show(res$0));$skip(10); val res$1 = 
  "ls".!!;System.out.println("""res1: String = """ + $show(res$1));$skip(71); val res$2 = 
  "ls /home/mort/workspace/dissolve-struct/data/generated/neuro/..".!!;System.out.println("""res2: String = """ + $show(res$2));$skip(135); 
  val pp= sys.process.Process(Seq("ls","-l"), new java.io.File("/home/mort/workspace/dissolve-struct/data/generated/neuro/Images/.."));System.out.println("""pp  : scala.sys.process.ProcessBuilder = """ + $show(pp ));$skip(11); val res$3 = 
  pp.lines;System.out.println("""res3: Stream[String] = """ + $show(res$3));$skip(8); val res$4 = 
  pp.!!;System.out.println("""res4: String = """ + $show(res$4));$skip(24); val res$5 = 
  
  "cd /home/mort".!!;System.out.println("""res5: String = """ + $show(res$5));$skip(68); val res$6 = 
  "cd /home/mort/workspace/dissolve-struct/data/generated/neuro".!!;System.out.println("""res6: String = """ + $show(res$6))}
}
