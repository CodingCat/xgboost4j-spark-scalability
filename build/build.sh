#!/bin/sh

TARGET_URL=https://github.com/codingcat/xgboost.git
TARGET_BRANCH=master2

# compile xgboost
rm -rf xgboost_build;
git clone --recursive $TARGET_URL xgboost_build; 
cd xgboost_build;

if [ -n $TARGET_BRANCH ]; then
  git fetch --all
  git checkout $TARGET_BRANCH;
fi

rm -rf lib/*
cd jvm-packages;
mvn package;
cp xgboost4j-spark/target/*-dependencies.jar ../../lib/

# compile benchmark
cd ../../;
build/sbt assembly
rm -rf xgboost_build
