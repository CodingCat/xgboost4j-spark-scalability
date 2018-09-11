#!/bin/bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR/../"

# clean already downloaded files

cd $PROJECT_DIR

if [ ! -d "KittenWhisker" ]; then
  # download kittenwhisker
  git clone git@github.com:CodingCat/KittenWhisker.git
  # run the script to compile KittenWhisker 
  cd KittenWhisker

  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    mvn package 
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    . dev/build-docker.sh
  fi
fi

# update command to start benchmark workload
cd $PROJECT_DIR;
cat > KittenWhisker/spark_app_cmd.sh <<'EOT'
spark-submit --class me.codingcat.xgboost4j.AirlineClassifier --num-executors 10 --executor-memory 14g --executor-cores 8 --driver-memory 14g --driver-cores 4 --files conf/airline.conf --master yarn-cluster --queue hadoop-adhoc target/scala-2.11/xgboost4j-spark-scalability-assembly-0.1-SNAPSHOT.jar ./airline.conf
EOT
