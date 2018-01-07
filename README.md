# xgboost4j-spark-scalability

a benchmark to test scalability of xgboost4j-spark and relevant projects

## Prerequestes 

You have to ensure that maven (3.0+) and cmake is installed in your $PATH

## Build Benchmark

1. Edit build/build.sh and define variables like TARGET_URL, TARGET_BRANCH

2. run build/build.sh

3. You get the benchmark jar in target/ 


## Run Benchmarks

1. Generate Data:

```bash
spark-submit --master yarn-cluster --num-executors 10 --executor-memory 6g --executor-cores 8 \
    --class me.codingcat.xgboost4j.AirlineDataGenerator --files conf/airline_datagen.conf \
     target/scala-2.11/xgboost4j-spark-scalability-assembly-0.1-SNAPSHOT.jar ./airline_datagen.conf
```

2. Run workload:

```bash
spark-submit --master yarn-cluster --num-executors 10 --executor-memory 6g --executor-cores 8 \
    --class me.codingcat.xgboost4j.AirlineClassifier --files conf/airline.conf \
     target/scala-2.11/xgboost4j-spark-scalability-assembly-0.1-SNAPSHOT.jar ./airline.conf
```