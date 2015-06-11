package ch.ethz.dalab.dissolve.examples.neighbourhood

object funfun {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(114); 
  println("Welcome to the Scala worksheet");$skip(138); 
  
  
  
  
  def aFn (somearg :Int , maybearg:Int = 100): Int={
  		if(maybearg==100)
  			return 100;
  			else
  			return somearg
  };System.out.println("""aFn: (somearg: Int, maybearg: Int)Int""");$skip(20); 
  
  val t = aFn(5);System.out.println("""t  : Int = """ + $show(t ));$skip(12); val res$0 = 
  aFn(5,10);System.out.println("""res0: Int = """ + $show(res$0));$skip(18); 
  val nein = None;System.out.println("""nein  : None.type = """ + $show(nein ))}
  
  
}
