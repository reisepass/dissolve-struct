#!/bin/bash
wget https://dl.bintray.com/sbt/native-packages/sbt/0.13.8/sbt-0.13.8.tgz
tar -xvf sbt-0.13.8.tgz 
cd dissolve-struct-lib/
~/sbt/bin/sbt publish-local
cd ../dissolve-struct-examples
~/sbt/bin/sbt compile
~/sbt/bin/sbt package
