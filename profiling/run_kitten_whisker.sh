#!/bin/bash

set -x

# download kittenwhisker
wget https://github.com/CodingCat/KittenWhisker/archive/release-0.1.tar.gz

# uncompress tar ball
tar -zxvf release-0.1.tar.gz

# run the script to start xgboost4j-spark benchmark workload

cd KittenWhisker-release-0.1;
. dev/build-docker.sh
