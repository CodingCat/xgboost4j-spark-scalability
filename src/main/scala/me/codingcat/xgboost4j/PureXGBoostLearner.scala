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

import scala.collection.mutable.ListBuffer
import scala.io.Source

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostRegressor}

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.SparkSession

object PureXGBoostLearner {

  def main(args: Array[String]): Unit = {
    val featureCol = args(0)
    val labelCol = args(1)
    val inputPath = args(2)
    val trainingRatio = args(3).toDouble
    val isRegression = args(4).toBoolean
    val configFile = args(5)
    val modelOutputPath = args(6)

    val xgbParamMap = {
      val lb = new ListBuffer[(String, String)]
      for (line <- Source.fromFile(configFile).getLines()) {
        val array = line.split("=")
        lb += array(0) -> array(1)
      }
      lb.toMap
    }

    val columnsOtherThanFeature = if (xgbParamMap.contains("group_col")) {
      Seq(xgbParamMap("group_col"), labelCol)
    } else {
      Seq(labelCol)
    }

    val spark = SparkSession.builder().getOrCreate()
    val Array(trainingSet, testSet) = spark.read.parquet(inputPath).
      select(featureCol, columnsOtherThanFeature: _*).
      randomSplit(Array(trainingRatio, 1 - trainingRatio))

    val xgbLearner = if (isRegression) {
      new XGBoostRegressor(xgbParamMap)
    } else {
      new XGBoostClassifier(xgbParamMap)
    }
    if (trainingRatio < 1) {
      xgbLearner.setEvalSets(Map("test" -> testSet))
    }
    xgbLearner.setFeaturesCol(featureCol)
    xgbLearner.setLabelCol(labelCol)

    val startTS = System.currentTimeMillis()
    val xgbModel = xgbLearner.fit(trainingSet)
    println(s"finished training in ${System.currentTimeMillis() - startTS}")
    xgbModel.write.overwrite().save(modelOutputPath)
  }
}
