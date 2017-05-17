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
import org.apache.spark.ml.classification.GBTClassifier
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
      xgbEstimator: XGBoostEstimator,
      trainingSet: DataFrame,
      tuningParamsPath: String): XGBoostModel = {
    val conf = ConfigFactory.parseFile(new File(tuningParamsPath))
    val paramGrid = new ParamGridBuilder()
      .addGrid(xgbEstimator.eta, Utils.fromConfigToParamGrid(conf)(xgbEstimator.eta.name))
      .addGrid(xgbEstimator.maxDepth, Utils.fromConfigToParamGrid(conf)(xgbEstimator.maxDepth.name).
        map(_.toInt))
      .addGrid(xgbEstimator.gamma, Utils.fromConfigToParamGrid(conf)(xgbEstimator.gamma.name))
      .addGrid(xgbEstimator.lambda, Utils.fromConfigToParamGrid(conf)(xgbEstimator.lambda.name))
      .addGrid(xgbEstimator.colSampleByTree, Utils.fromConfigToParamGrid(conf)(
        xgbEstimator.colSampleByTree.name))
      .addGrid(xgbEstimator.subSample, Utils.fromConfigToParamGrid(conf)(
        xgbEstimator.subSample.name))
      .build()
    val cv = new CrossValidator()
      .setEstimator(xgbEstimator)
      .setEvaluator(new BinaryClassificationEvaluator().
        setRawPredictionCol("probabilities").setLabelCol("label"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
    val cvModel = cv.fit(trainingSet)
    cvModel.bestModel.asInstanceOf[XGBoostModel]
  }

  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.parseFile(new File(args(0)))
    val libName = config.getString("me.codingcat.xgboost4j.lib")
    val trainingPath = config.getString("me.codingcat.xgboost4j.airline.trainingPath")
    val trainingRounds = config.getInt("me.codingcat.xgboost4j.rounds")
    val numWorkers = config.getInt("me.codingcat.xgboost4j.numWorkers")
    val treeType = config.getString("me.codingcat.xgboost4j.treeMethod")
    val sampleRate = config.getDouble("me.codingcat.xgboost4j.sampleRate")
    val params = Utils.fromConfigToXGBParams(config)
    val spark = SparkSession.builder().getOrCreate()
    val completeSet = spark.read.parquet(trainingPath)
    val sampledDataset = if (sampleRate > 0) {
      completeSet.sample(withReplacement = false, sampleRate)
    } else {
      completeSet
    }
    val Array(trainingSet, testSet) = sampledDataset.randomSplit(Array(0.8, 0.2))

    val pipeline = buildPreprocessingPipeline()
    val transformedTrainingSet = runPreprocessingPipeline(pipeline, trainingSet)
    val transformedTestset = runPreprocessingPipeline(pipeline, testSet)

    if (libName == "xgboost") {
      if (args.length >= 2) {
        val xgbEstimator = new XGBoostEstimator(params)
        xgbEstimator.set(xgbEstimator.round, trainingRounds)
        xgbEstimator.set(xgbEstimator.nWorkers, numWorkers)
        xgbEstimator.set(xgbEstimator.treeMethod, treeType)
        val bestModel = crossValidationWithXGBoost(xgbEstimator, transformedTrainingSet, args(1))
        println(s"best model: ${bestModel.extractParamMap()}")
        val eval = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
        println("eval results: " + eval.evaluate(bestModel.transform(transformedTestset)))
      } else {
        // directly training
        transformedTrainingSet.cache().foreach(_ => Unit)
        val startTime = System.nanoTime()
        val xgbModel = XGBoost.trainWithDataFrame(transformedTrainingSet, round = trainingRounds,
          nWorkers = numWorkers, params = Utils.fromConfigToXGBParams(config))
        println(s"===training time cost: ${(System.nanoTime() - startTime) / 1000.0 / 1000.0} ms")
        val resultDF = xgbModel.transform(transformedTestset)
        val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
        binaryClassificationEvaluator.setRawPredictionCol("probabilities").setLabelCol("label")
        println(s"=====test AUC: ${binaryClassificationEvaluator.evaluate(resultDF)}======")
      }
    } else {
      val gradientBoostedTrees = new GBTClassifier()
      gradientBoostedTrees.setMaxBins(1000)
      gradientBoostedTrees.setMaxIter(20)
      gradientBoostedTrees.setMaxDepth(7)
      val model = gradientBoostedTrees.fit(transformedTrainingSet)
      val eval = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
      println("eval results: " + eval.evaluate(model.transform(transformedTrainingSet)))
    }
  }
}
