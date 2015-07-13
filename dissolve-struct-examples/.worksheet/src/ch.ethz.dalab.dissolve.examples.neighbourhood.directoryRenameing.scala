package ch.ethz.dalab.dissolve.examples.neighbourhood
import java.io._
import scala.io.Source
import java.util.HashMap
object directoryRenameing {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(190); 
  println("Welcome to the Scala worksheet");$skip(87); 
  
  
  val baseDir = "/home/mort/workspace/dissolve-struct/dissolve-struct-examples/";System.out.println("""baseDir  : String = """ + $show(baseDir ));$skip(84); 
  
     val lotsOfDir = new File(baseDir).list;System.out.println("""lotsOfDir  : Array[String] = """ + $show(lotsOfDir ));$skip(55);  //.map(_.filter(_.endsWith(".bmp")))
     
     
     val hashMap = new HashMap[Int,Double];System.out.println("""hashMap  : java.util.HashMap[Int,Double] = """ + $show(hashMap ));$skip(394); 
     for ( i <- 0 until lotsOfDir.size){
     		  if(lotsOfDir(i).contains("dgenD_AA")){
     		  val dGenFCheck = new File(baseDir+lotsOfDir(i)+"/dataGenerationConfig.cfg")
     			val dGenF = new FileReader(baseDir+lotsOfDir(i)+"/dataGenerationConfig.cfg")
     			if(dGenFCheck.exists){
	     			val tt= new BufferedReader(dGenF);
	     		  
	     		  
     			}
     			
     			
     			}
     
     };$skip(37); 
     println(lotsOfDir)}
     
    
}
