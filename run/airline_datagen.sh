#!/bin/sh

NUM_EXECUTORS=6
EXECUTOR_CORES=8
EXECUTOR_MEMORY=8g

spark-submit --master yarn-cluster --num-executors $NUM_EXECUTORS --executor-memory $EXECUTOR_MEMORY --executor-cores $EXECUTOR_CORES \
    --class me.codingcat.xgboost4j.AirlineDataGenerator --files conf/airline_datagen.conf \
     target/scala-2.11/xgboost4j-spark-scalability-assembly-0.1-SNAPSHOT.jar ./airline_datagen.conf
