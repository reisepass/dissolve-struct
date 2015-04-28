package ch.ethz.dalab.dissolve.examples.neighbourhood

import sys.process._
/**
 * @author mort
 */
object bashExperiments {
  def main(args: Array[String]): Unit = {

      //Experiment for comparing pairwise vs justUnary. Find out how many iterations are needed to mach accuracy
      
      val optionsRange = Map( "-samplefrac" -> Set("0.2","1"),
                              "-onlyUnary" -> Set("true", "false"))
      val baseDissolveCommand = 
        """~/spark/bin/spark-submit \
--class ch.ethz.dalab.dissolve.examples.neighbourhood.runMSRC \
--jars "/root/.ivy2/local/ch.ethz.dalab/dissolvestruct_2.10/0.1-SNAPSHOT/jars/dissolvestruct_2.10.jar,/root/.ivy2/cache/cc.factorie/factorie/jars/factorie-1.0.jar,/root/.ivy2/cache/com.github.scopt/scopt_2.10/jars/scopt_2.10-3.3.0.jar" \
--master local \
--driver-memory 2G \
target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar"""  
      
      val firstExec:String = baseDissolveCommand + "-samplefrac=0.2 -onlyUnary=true"
      val printOuts = firstExec!!
      
      println(printOuts)
      
      
      }
}