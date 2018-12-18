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

import scala.collection.mutable.ListBuffer
import scala.io.Source

import com.typesafe.config.ConfigFactory
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor

import org.apache.spark.sql.SparkSession

object PureClassifier {

  def main(args: Array[String]): Unit = {
    val featureCol = args(0)
    val labelCol = args(1)
    val inputPath = args(2)
    val ratio = args(3).toDouble
    val configFile = args(4)

    val xgbParamMap = {
      val lb = new ListBuffer[(String, String)]
      for (line <- Source.fromFile(configFile).getLines()) {
        val array = line.split("=")
        lb += array(0) -> array(1)
      }
      lb.toMap
    }

    val spark = SparkSession.builder().getOrCreate()
    val trainingSet = spark.read.parquet(inputPath).select(featureCol, labelCol).sample(ratio)

    val xgbRegressor = new XGBoostRegressor(xgbParamMap)
    xgbRegressor.setFeaturesCol(featureCol)
    xgbRegressor.setLabelCol(labelCol)

    val startTS = System.currentTimeMillis()
    val xgbRegressionModel = xgbRegressor.fit(trainingSet)
    println(s"finished training in ${System.currentTimeMillis() - startTS}")
  }
}
