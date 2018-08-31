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

import com.typesafe.config.ConfigFactory
import me.codingcat.xgboost4j.common.Utils
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

object AirlineClassifier {

  private def buildPreprocessingPipeline(): Pipeline = {
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
        "encodedCarrier", "encodedOrigin", "encodedDest", "Distance")
    ).setOutputCol("features")
    val pipeline = new Pipeline().setStages(
      Array(monthIndexer, daysOfMonthIndexer, daysOfWeekIndexer,
        uniqueCarrierIndexer, originIndexer, destIndexer, monthEncoder, daysOfMonthEncoder,
        daysOfWeekEncoder, uniqueCarrierEncoder, originEncoder, destEncoder, vectorAssembler))
    pipeline
  }

  private def runPreprocessingPipeline(pipeline: Pipeline, trainingSet: DataFrame): DataFrame = {
    pipeline.fit(trainingSet).transform(trainingSet).selectExpr(
      "features", "case when dep_delayed_15min = true then 1.0 else 0.0 end as label")
  }

  private def crossValidationWithXGBoost(
      xgbClassifier: XGBoostClassifier,
      trainingSet: DataFrame,
      tuningParamsPath: String): XGBoostClassificationModel = {
    val conf = ConfigFactory.parseFile(new File(tuningParamsPath))
    val paramGrid = new ParamGridBuilder()
      .addGrid(xgbClassifier.eta, Utils.fromConfigToParamGrid(conf)(xgbClassifier.eta.name))
      .addGrid(xgbClassifier.maxDepth, Utils.fromConfigToParamGrid(conf)(
        xgbClassifier.maxDepth.name).map(_.toInt))
      .addGrid(xgbClassifier.gamma, Utils.fromConfigToParamGrid(conf)(xgbClassifier.gamma.name))
      .addGrid(xgbClassifier.lambda, Utils.fromConfigToParamGrid(conf)(xgbClassifier.lambda.name))
      .addGrid(xgbClassifier.colsampleBytree, Utils.fromConfigToParamGrid(conf)(
        xgbClassifier.colsampleBytree.name))
      .addGrid(xgbClassifier.subsample, Utils.fromConfigToParamGrid(conf)(
        xgbClassifier.subsample.name))
      .build()
    val cv = new CrossValidator()
      .setEstimator(xgbClassifier)
      .setEvaluator(new BinaryClassificationEvaluator().
        setRawPredictionCol("probabilities").setLabelCol("label"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
    val cvModel = cv.fit(trainingSet)
    cvModel.bestModel.asInstanceOf[XGBoostClassificationModel]
  }

  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.parseFile(new File(args(0)))
    val trainingPath = config.getString("me.codingcat.xgboost4j.airline.trainingPath")
    val trainingRounds = config.getInt("me.codingcat.xgboost4j.rounds")
    val numWorkers = config.getInt("me.codingcat.xgboost4j.numWorkers")
    val treeType = config.getString("me.codingcat.xgboost4j.treeMethod")
    val params = Utils.fromConfigToXGBParams(config)
    val spark = SparkSession.builder().getOrCreate()
    val dataSet = spark.read.parquet(trainingPath)
    val Array(trainingSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))

    val pipeline = buildPreprocessingPipeline()
    val transformedTrainingSet = runPreprocessingPipeline(pipeline, trainingSet)
    val transformedTestset = runPreprocessingPipeline(pipeline, testSet)

    if (args.length >= 2) {
      val xgbClassifier = new XGBoostClassifier(params)
      xgbClassifier.set(xgbClassifier.numRound, trainingRounds)
      xgbClassifier.set(xgbClassifier.numWorkers, numWorkers)
      xgbClassifier.set(xgbClassifier.treeMethod, treeType)
      val bestModel = crossValidationWithXGBoost(xgbClassifier, transformedTrainingSet, args(1))
      println(s"best model: ${bestModel.extractParamMap()}")
      val eval = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
      println("eval results: " + eval.evaluate(bestModel.transform(transformedTestset)))
    } else {
      // directly training
      transformedTrainingSet.cache().foreach(_ => Unit)
      val startTime = System.nanoTime()
      val xgbParams = Utils.fromConfigToXGBParams(config)
      val xgboostClassifier = new XGBoostClassifier(xgbParams)
      val xgboostClassificationModel = xgboostClassifier.fit(transformedTrainingSet)
      println(s"===training time cost: ${(System.nanoTime() - startTime) / 1000.0 / 1000.0} ms")
      val resultDF = xgboostClassificationModel.transform(transformedTestset)
      val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
      binaryClassificationEvaluator.setRawPredictionCol("probabilities").setLabelCol("label")
      println(s"=====test AUC: ${binaryClassificationEvaluator.evaluate(resultDF)}======")
    }
  }
}
