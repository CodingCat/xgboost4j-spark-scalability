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

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Row, SaveMode, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object CriteoDataGenerator {

  private def buildPipeline(): Pipeline = {
    val pipeline = new Pipeline()
    val stringIndexerArray = new Array[StringIndexer](26)
    for (i <- 0 until 26) {
      stringIndexerArray(i) = new StringIndexer().setInputCol(s"category_$i").setOutputCol(
        s"category_$i" + "_index")
    }
    /*
    val oneHotEncodersArray = new Array[OneHotEncoder](26)
    for (i <- 0 until 26) {
      oneHotEncodersArray(i) = new OneHotEncoder().setInputCol(s"category_$i" + "_index").
        setOutputCol(s"category_$i" + "_encoded")
    }
    */
    val numericColumns = (0 until 13).map(i => s"numeric_$i").toArray
    val categoryColumns = stringIndexerArray.map(_.getOutputCol)
    val vectorAssembler = new VectorAssembler().setInputCols(numericColumns ++ categoryColumns).
      setOutputCol("features")
    /* ++ oneHotEncodersArray */
    pipeline.setStages(stringIndexerArray :+ vectorAssembler)
  }

  def main(args: Array[String]): Unit = {
    val trainingInputPath = args(0)
    val outputPath = args(1)
    val partitions = args(2).toInt
    val spark = SparkSession.builder().getOrCreate()
    val df = spark.read.format("csv").option("delimiter", "\t").load(trainingInputPath).
      repartition(partitions)
    val df2 = df.toDF(Seq("label") ++ (0 until 13).map(i => s"numeric_$i") ++
      (0 until 26).map(i => s"category_$i"): _*)
    val handledNull = df2.na.fill("NONE", (0 until 26).map(i => s"category_$i")).
      na.fill(Double.NaN, (0 until 13).map(i => s"numeric_$i"))
    val castExprArray = (0 until 13).map(i => s"cast (numeric_$i as double) numeric_$i")
    val remainExprArray = (0 until 26).map(i => s"category_$i")
    val typeTransformedDF = handledNull.selectExpr(
      Seq("cast (label as double) label") ++ castExprArray ++ remainExprArray: _*)
    /*
    val pipeline = buildPipeline()
    val transformedDF =
      pipeline.fit(typeTransformedDF).transform(typeTransformedDF).select("features", "label")
    */
    typeTransformedDF.write.format("parquet").mode(SaveMode.Append).save(outputPath)
  }
}
