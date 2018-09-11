#!/bin/bash

set -x

# download kittenwhisker
wget https://github.com/CodingCat/KittenWhisker/archive/release-0.1.tar.gz

# uncompress tar ball
tar -zxvf release-0.1.tar.gz

# run the script to start xgboost4j-spark benchmark workload

cd KittenWhisker-release-0.1;
. dev/build-docker.sh
cd -;
cat > KittenWhisker-release-0.1/spark_app_cmd.sh <<'EOT'
spark-submit --class me.codingcat.xgboost4j.AirlineClassifier --num-executors 10 --executor-memory 14g --executor-cores 8 --driver-memory 14g --driver-cores 4 --files conf/airline.conf --master yarn-cluster --queue hadoop-adhoc ../target/scala-2.11/xgboost4j-spark-scalability-assembly-0.1-SNAPSHOT.jar ./airline.conf
EOT
