#!/usr/bin/env bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR/../"

# compile XGBoost

cd $PROJECT_DIR;
rm -rf xgboost_upstream;
git clone --recursive git@github.com:CodingCat/xgboost.git xgboost_upstream
cd $PROJECT_DIR/xgboost_upstream/jvm-packages; git checkout instrumentation

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    mvn package -DskipTests; 
elif [[ "$OSTYPE" == "darwin"* ]]; then
    . dev/build-docker.sh
fi

# mv jars

mkdir -p $PROJECT_DIR/lib;
cp -v $PROJECT_DIR/xgboost_upstream/jvm-packages/xgboost4j/target/xgboost4j-* $PROJECT_DIR/lib;
cp -v $PROJECT_DIR/xgboost_upstream/jvm-packages/xgboost4j-spark/target/xgboost4j-* $PROJECT_DIR/lib;
