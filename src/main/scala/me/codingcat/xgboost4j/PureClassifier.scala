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
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor

import org.apache.spark.sql.SparkSession

object PureClassifier {

  def main(args: Array[String]): Unit = {
    val featureCol = args(0)
    val labelCol = args(1)
    val inputPath = args(2)
    val ratio = args(3).toDouble
    val configFile = args(4)

    val conf = ConfigFactory.parseFile(new File(configFile))
    import scala.collection.JavaConverters._
    val xgbParamMap = conf.entrySet().asScala.map {
      entry =>
        entry.getKey -> conf.getString(entry.getKey)
    }.toMap

    val spark = SparkSession.builder().getOrCreate()
    val trainingSet = spark.read.parquet(inputPath).select(featureCol, labelCol).sample(ratio)

    val xgbRegressor = new XGBoostRegressor(xgbParamMap)

    val startTS = System.currentTimeMillis()
    val xgbRegressionModel = xgbRegressor.fit(trainingSet)
    println(s"finished training in ${System.currentTimeMillis() - startTS}")
  }
}
