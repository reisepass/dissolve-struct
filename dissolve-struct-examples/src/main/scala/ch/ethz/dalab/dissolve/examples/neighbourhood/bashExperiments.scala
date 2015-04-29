package ch.ethz.dalab.dissolve.examples.neighbourhood

import sys.process._
/**
 * @author mort
 */
object bashExperiments {
  def main(args: Array[String]): Unit = {

      val isEC2 = false 
      //Experiment for comparing pairwise vs justUnary. Find out how many iterations are needed to mach accuracy
      
      val optionsRange = Map( "-samplefrac" -> Set("0.2","1"),
                              "-onlyUnary"  -> Set("true", "false"))
      "mkdir experiment1"!           
      
      val baseDissolveCommand = if(isEC2) {
        "/root/spark/bin/spark-submit --class ch.ethz.dalab.dissolve.examples.neighbourhood.runMSRC --jars \"/root/.ivy2/local/ch.ethz.dalab/dissolvestruct_2.10/0.1-SNAPSHOT/jars/dissolvestruct_2.10.jar,/root/.ivy2/cache/cc.factorie/factorie/jars/factorie-1.0.jar,/root/.ivy2/cache/com.github.scopt/scopt_2.10/jars/scopt_2.10-3.3.0.jar\" --master local --driver-memory 2G  target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar"
      }
      else {
        """ /home/mort/Downloads/spark-1.3.1-bin-hadoop2.6/bin/spark-submit \
 --class ch.ethz.dalab.dissolve.examples.neighbourhood.runMSRC \
 --jars "/home/mort/.ivy2/local/ch.ethz.dalab/dissolvestruct_2.10/0.1-SNAPSHOT/jars/dissolvestruct_2.10.jar,/home/mort/.ivy2/cache/cc.factorie/factorie/jars/factorie-1.0.jar,/home/mort/.ivy2/cache/com.github.scopt/scopt_2.10/jars/scopt_2.10-3.3.0.jar" \
 --master local \
  --driver-memory 2G \
    target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar -local -onlyUnary
    """
      }
      
      val baseDissolveCommandL=  """/home/mort/Downloads/spark-1.3.1-bin-hadoop2.6/bin/spark-submit \
 --class ch.ethz.dalab.dissolve.examples.neighbourhood.runMSRC \
 --jars "/home/mort/.ivy2/local/ch.ethz.dalab/dissolvestruct_2.10/0.1-SNAPSHOT/jars/dissolvestruct_2.10.jar,/home/mort/.ivy2/cache/cc.factorie/factorie/jars/factorie-1.0.jar,/home/mort/.ivy2/cache/com.github.scopt/scopt_2.10/jars/scopt_2.10-3.3.0.jar" \
 --master local \
  --driver-memory 2G \
    target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar -local -onlyUnary
    """
      
      
      val firstExec:String = baseDissolveCommand.concat( 
          " -samplefrac=0.2 -onlyUnary=true ").concat(" > /root/experiment1/exp1Log")
      
       val otherExec = java.net.URLEncoder.encode(firstExec, "UTF-8");
     
      val printOuts = firstExec!!
      
      println(printOuts)
      
      
      }
}