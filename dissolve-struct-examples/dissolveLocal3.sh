#!/bin/bash

EXP_NAME=LocalMF_parSpace_2
logOut=${EXP_NAME}.log
csvOut=${EXP_NAME}.csv

baseCMD="/home/mort/programs/spark-1.3.1-bin-hadoop2.6/bin/spark-submit --class ch.ethz.dalab.dissolve.examples.neighbourhood.runMSRC --jars /home/mort/.ivy2/local/ch.ethz.dalab/dissolvestruct_2.10/0.1-SNAPSHOT/jars/dissolvestruct_2.10.jar,/home/mort/.ivy2/cache/cc.factorie/factorie/jars/factorie-1.0.jar,/home/mort/.ivy2/cache/com.github.scopt/scopt_2.10/jars/scopt_2.10-3.3.0.jar --driver-memory 2G     target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar -local=true -debug=false"
    

echo $baseCMD

touch $logOut
${baseCMD} -runName=${EXP_NAME}_1 -onlyUnary  useMF=false roundLimit=1   dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_2             useMF=false roundLimit=1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 &>> $logOut

#Testing Unary vs Pairwie effectivness
${baseCMD} -runName=${EXP_NAME}_3 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=1   &>> $logOut
${baseCMD} -runName=${EXP_NAME}_4             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=1 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_5 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=2 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_6             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=2 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_7 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_8             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_9 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3  dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=4 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_10             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=4  &>> $logOut
${baseCMD} -runName=${EXP_NAME}_11 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=5  &>> $logOut
${baseCMD} -runName=${EXP_NAME}_12             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=5  &>> $logOut
${baseCMD} -runName=${EXP_NAME}_13 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=6  &>> $logOut
${baseCMD} -runName=${EXP_NAME}_14             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=6 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_15 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=7 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=7 &>> $logOut

#more noise
${baseCMD} -runName=${EXP_NAME}_15 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=8 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=8 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_15 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=9 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=9 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_15 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=10 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=10 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_15 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=11 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=11 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_15 -onlyUnary  useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=12 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=false roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=12 &>> $logOut



#MF 
${baseCMD} -runName=${EXP_NAME}_4             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=1 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_6             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=2 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_8             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=3 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_10             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=4  &>> $logOut
${baseCMD} -runName=${EXP_NAME}_12             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=5  &>> $logOut
${baseCMD} -runName=${EXP_NAME}_14             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=6 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=7 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_16             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=8 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=9 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=10 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=11 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_16             useMF=true roundLimit=10  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.4 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=12 &>> $logOut

# MF Paramater, MaxIter, RoundLimit, Learning Rate
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16  dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16  dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16  dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16  dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10    learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10    learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10    learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=13 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10    learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=15 &>> $logOut




${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16  dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16  dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.1  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut


${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10    learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.05  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=50 roundLimit=10    learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=500 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=100 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=10 roundLimit=10    learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=60 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=200 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20             useMF=true maxDecodeItr=70 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=18 &>> $logOut





# Dif Num rounds
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=5  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=15  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=20  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=25  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=5  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=15  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=20  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=25  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=5  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=15  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=20  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=25  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=5  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=15  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=20  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=25  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=30 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut


#More data 
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=5  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=15  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=20  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=25  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=19 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=5  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=15  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=20  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=25  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=20 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=5  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=15  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=20  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=25  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=30 &>> $logOut

${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=5  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=10  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60  dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=15  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=20  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut
${baseCMD} -runName=${EXP_NAME}_20   useMF=true maxDecodeItr=30 roundLimit=25  learningRate=0.2  dataGenSparsity=0.7 numClasses=3 dataAddedNoise=0.3 dataNoiseOnlyTest=false dataGenTrainSize=60 dataGenCanvasSize=16 dataRandSeed=40 &>> $logOut




if ! [[ -e $csvOut ]] ; then
    #solverOptions.startTime, solverOptions.runName,solverOptions.gitVersion,(t1MTrain-t0MTrain),solverOptions.dataGenSparsity,solverOptions.dataAddedNoise,if(solverOptions.dataNoiseOnlyTest)"t"else"f",solverOptions.dataGenTrainSize,solverOptions.dataGenCanvasSize,solverOptions.learningRate,if(solverOptions.useMF)"t"else"f",solverOptions.numClasses,MAX_DECODE_ITERATIONS,if(solverOptions.onlyUnary)"t"else"f",if(solverOptions.debug)"t"else"f",solverOptions.roundLimit,if(solverOptions.dataWasGenerated)"t"else"f",avgTestLoss,avgTrainLoss    ) ) 
	echo -e 'label,startTime,runName,gitVersion,elapsedTime,dataGenSparsity,dataAddedNoise,dataNoiseOnlyTest,dataGenTrainSize,dataGenCanvasSize,learningRate,useMF,numClasses,MAX_DECODE_ITERATIONS,onlyUnary,debug,roundLimit,dataWasGenerated,avgTestLoss,avgTrainLoss,dataRandSeed\n'  > $csvOut
fi

cat ${logOut} | grep "#RoundProgTag#" >> $csvOut


