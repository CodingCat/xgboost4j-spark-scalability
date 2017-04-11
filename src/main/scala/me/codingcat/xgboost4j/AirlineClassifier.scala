/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package me.codingcat.xgboost4j

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}
import me.codingcat.xgboost4j.common.Utils
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostEstimator}

import org.apache.spark.sql.SparkSession

object AirlineClassifier {

  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.parseFile(new File(args(0)))
    val trainingPath = config.getString("me.codingcat.xgboost4j.airline.trainingPath")
    val trainingRounds = config.getInt("me.codingcat.xgboost4j.rounds")
    val numWorkers = config.getInt("me.codingcat.xgboost4j.numWorkers")
    val params = Utils.fromConfigToXGBParams(config)
    val spark = SparkSession.builder().getOrCreate()
    val trainingSet = spark.read.parquet(trainingPath)



    val xgbModel = XGBoost.trainWithDataFrame(trainingSet,
      params = params, round = trainingRounds, nWorkers = numWorkers)

    // TODO: evaluation part
    println(xgbModel.parent.asInstanceOf[XGBoostEstimator].uid)
  }
}
