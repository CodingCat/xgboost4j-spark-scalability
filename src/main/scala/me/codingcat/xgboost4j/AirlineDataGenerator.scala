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

import com.typesafe.config.ConfigFactory

import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._


object AirlineDataGenerator {

  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.builder().getOrCreate()

    val config = ConfigFactory.parseFile(new File(args(0)))
    val inputFile = config.getString("me.codingcat.xgboost4j.dataset.airline.inputPath")
    val ratioRate = config.getDouble("me.codingcat.xgboost4j.dataset.airline.sampleRate")
    val outputDir = config.getString("me.codingcat.xgboost4j.dataset.airline.outputDir")
    val mergedDF = sparkSession.read.option("header", "true").csv(inputFile)
    val dfWithNewCol = mergedDF.filter("DepDelay != 'NA'").withColumn("dep_delayed_15min", udf(
      (depDelay: String) => if (depDelay.toInt >= 15) true else false).apply(col("DepDelay")))
    val extractedDF = dfWithNewCol.select("Month", "DayofMonth", "DayOfWeek", "DepTime",
      "UniqueCarrier", "Origin", "Dest", "Distance", "dep_delayed_15min")
    val convertedDF = extractedDF.selectExpr("Month", "DayOfMonth", "DayOfWeek",
      "CAST(DepTime As Float) as DepTime",
      "UniqueCarrier", "Origin", "Dest", "CAST(Distance As Float) as Distance", "dep_delayed_15min")
    val sampledDF = convertedDF.sample(withReplacement = false, ratioRate)
    sampledDF.write.mode(SaveMode.Overwrite).parquet(outputDir)
  }
}
