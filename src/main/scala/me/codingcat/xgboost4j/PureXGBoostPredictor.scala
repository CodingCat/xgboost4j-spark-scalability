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

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostRegressionModel}

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.param
import org.apache.spark.sql.SparkSession

object PureXGBoostPredictor {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    val modelPath = args(0)
    val inputPath = args(1)
    val replicationFactor = args(2).toDouble
    val useExternalMemory = args(3).toBoolean
    var taskType = ""
    val xgbModel = {
      try {
        val model = XGBoostClassificationModel.load(modelPath)
        model.set(model.useExternalMemory, true)
        taskType = "classification"
        model
      } catch {
        case _: IllegalArgumentException =>
          val model = XGBoostRegressionModel.load(modelPath)
          model.set(model.useExternalMemory, true)
          taskType = "regression"
          model
      }
    }
    val inputDF = spark.read.parquet(inputPath)
    var finalDF = inputDF
    if (replicationFactor > 1) {
      for (i <- 1 until replicationFactor.toInt) {
        finalDF = finalDF.union(inputDF)
      }
    } else {
      finalDF = inputDF.sample(replicationFactor)
    }
    val metrics = if (taskType == "regression") {
      new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol(xgbModel.getLabelCol)
        .evaluate(xgbModel.transform(finalDF))
    } else if (taskType == "classification") {
      val numOfCls = xgbModel.asInstanceOf[XGBoostClassificationModel].numClasses
      if (numOfCls == 2) {
        new BinaryClassificationEvaluator()
          .setMetricName("areaUnderROC")
          .setLabelCol(xgbModel.getLabelCol)
          .evaluate(xgbModel.transform(finalDF))
      } else {
        new MulticlassClassificationEvaluator()
          .setMetricName("f1")
          .setLabelCol(xgbModel.getLabelCol)
          .evaluate(xgbModel.transform(finalDF))
      }
    }
    println(s"evaluation metrics: $metrics")
  }
}
