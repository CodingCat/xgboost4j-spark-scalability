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

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

import org.apache.spark.sql.SparkSession

object SampleApp {
  def main(): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    val df_sample = spark.read.parquet("/user/nanzhu/sample_dataset")
    val paramMapXgb8 = Map(
      "eta" -> 0.05f,
      "max_depth" -> 6,
      "objective" -> "binary:logistic",
      "colsample_bytree" -> 0.3,
      "num_round" -> 1000,
      "num_workers" -> 100,
      "nthread" -> 4)

    val xgbClassifier = new XGBoostClassifier(paramMapXgb8)
      .setFeaturesCol("features")
      .setLabelCol("label")

    val xgbClassificationModel = xgbClassifier.fit(df_sample)
  }
}
