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
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostEstimator, XGBoostModel}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object AirlineClassifier {

  private def buildPipeline(): Pipeline = {
    // string indexers
    val monthIndexer = new StringIndexer().setInputCol("Month").setOutputCol("monthIdx")
    val daysOfMonthIndexer = new StringIndexer().setInputCol("DayOfMonth").
      setOutputCol("dayOfMonthIdx")
    val daysOfWeekIndexer = new StringIndexer().setInputCol("DayOfWeek").
      setOutputCol("daysOfWeekIdx")
    val uniqueCarrierIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol(
      "uniqueCarrierIndex")
    val originIndexer = new StringIndexer().setInputCol("Origin").setOutputCol(
      "originIndexer")
    val destIndexer = new StringIndexer().setInputCol("Dest").setOutputCol(
      "destIndexer")
    // one-hot encoders
    val monthEncoder = new OneHotEncoder().setInputCol("monthIdx").
      setOutputCol("encodedMonth")
    val daysOfMonthEncoder = new OneHotEncoder().setInputCol("dayOfMonthIdx").
      setOutputCol("encodedDaysOfMonth")
    val daysOfWeekEncoder = new OneHotEncoder().setInputCol("daysOfWeekIdx").
      setOutputCol("encodedDaysOfWeek")
    val uniqueCarrierEncoder = new OneHotEncoder().setInputCol("uniqueCarrierIndex").
      setOutputCol("encodedCarrier")
    val originEncoder = new OneHotEncoder().setInputCol("originIndexer").
      setOutputCol("encodedOrigin")
    val destEncoder = new StringIndexer().setInputCol("destIndexer").setOutputCol(
      "encodedDest")

    val vectorAssembler = new VectorAssembler().setInputCols(
      Array("encodedMonth", "encodedDaysOfMonth", "encodedDaysOfWeek", "DepTime",
        "encodedCarrier", "encodedOrigin", "encodedDest", "Distance", "dep_delayed_15min")
    ).setOutputCol("features")
    val pipeline = new Pipeline().setStages(
      Array(monthIndexer, daysOfMonthIndexer, daysOfWeekIndexer,
        uniqueCarrierIndexer, originIndexer, destIndexer, monthEncoder, daysOfMonthEncoder,
        daysOfWeekEncoder, uniqueCarrierEncoder, originEncoder, destEncoder, vectorAssembler))
    pipeline
    /*
    pipeline.fit(inputDF).transform(inputDF).drop("workClass", "education", "maritalStatus",
      "occupation", "relationship", "race", "sex", "nativeCountry", "label")
    */
  }

  private def runPipeline(pipeline: Pipeline, trainingSet: DataFrame): DataFrame = {
    pipeline.fit(trainingSet).transform(trainingSet).select(
      "encodedMonth", "encodedDaysOfMonth", "encodedDaysOfWeek", "DepTime",
      "encodedCarrier", "encodedOrigin", "encodedDest", "Distance", "dep_delayed_15min"
    )
  }

  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.parseFile(new File(args(0)))
    val trainingPath = config.getString("me.codingcat.xgboost4j.airline.trainingPath")
    val trainingRounds = config.getInt("me.codingcat.xgboost4j.rounds")
    val numWorkers = config.getInt("me.codingcat.xgboost4j.numWorkers")
    val params = Utils.fromConfigToXGBParams(config)
    val spark = SparkSession.builder().getOrCreate()
    val trainingSet = spark.read.parquet(trainingPath)

    val pipeline = buildPipeline()
    val transformedTrainingSet = runPipeline(pipeline, trainingSet)
    val xgbModel = XGBoost.trainWithDataFrame(transformedTrainingSet,
      params = params, round = trainingRounds, nWorkers = numWorkers)
    xgbModel.transform(transformedTrainingSet).show()
  }
}
