#!/bin/bash

EXP_NAME=LocalSynthDataExp4_sprincleNoise
logOut=${EXP_NAME}.log
csvOut=${EXP_NAME}.csv

baseCMD="/home/mort/programs/spark-1.3.1-bin-hadoop2.6/bin/spark-submit --class ch.ethz.dalab.dissolve.examples.neighbourhood.runMSRC --jars /home/mort/.ivy2/local/ch.ethz.dalab/dissolvestruct_2.10/0.1-SNAPSHOT/jars/dissolvestruct_2.10.jar,/home/mort/.ivy2/cache/cc.factorie/factorie/jars/factorie-1.0.jar,/home/mort/.ivy2/cache/com.github.scopt/scopt_2.10/jars/scopt_2.10-3.3.0.jar --driver-memory 2G     target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar -local=false -debug=true"
    
if ! [[ -e $csvOut ]] ; then
	# " println("#RoundProgTag# , %s , %s , %.3f, %d, %f, %f, %f, %f, %f , %.2f, %s, %s"        .format(solverOptions.runName,solverOptions.gitVersion,elapsedTime, roundNum, dualityGap, primal,dual, trainError, testError,solverOptions.sampleFrac, if(solverOptions.doWeightedAveraging) "t" else "f", if(solverOptions.onlyUnary) "t" else "f"  )) "
       
	echo -e 'startTime,runName,gitVersion,elapsedTime,roundNum,dualityGap,primal,dual,trainError,testError,sampleFrac,doWeightedAveraging,onlyUnary\n'  > $csvOut
fi
echo $baseCMD

touch $logOut
${baseCMD} -runName=${EXP_NAME}_1 -onlyUnary -samplefrac=1 roundLimit=1   dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_7  -samplefrac=1 roundLimit=1 dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_2  -onlyUnary -samplefrac=1 roundLimit=10   dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_3 -onlyUnary -samplefrac=1 roundLimit=15  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_4  -onlyUnary -samplefrac=1 roundLimit=15   dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_5 -onlyUnary -samplefrac=1 roundLimit=15  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_6  -onlyUnary -samplefrac=1 roundLimit=15   dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 &>> $logOut


${baseCMD} -runName=${EXP_NAME}_8  -samplefrac=1 roundLimit=10 dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3&>> $logOut
${baseCMD} -runName=${EXP_NAME}_9  -samplefrac=1 roundLimit=10 dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3&>> $logOut
${baseCMD} -runName=${EXP_NAME}_10  -samplefrac=1 roundLimit=10 dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3&>> $logOut
${baseCMD} -runName=${EXP_NAME}_11  -samplefrac=1 roundLimit=10 dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3&>> $logOut
${baseCMD} -runName=${EXP_NAME}_12  -samplefrac=1 roundLimit=10 dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3&>> $logOut
${baseCMD} -runName=${EXP_NAME}_13  -samplefrac=1 roundLimit=10 dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3&>> $logOut

cat ${logOut} | grep "#RoundProgTag#" >> $csvOut


