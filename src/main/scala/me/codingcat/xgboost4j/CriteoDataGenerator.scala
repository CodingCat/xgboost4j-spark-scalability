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
import org.apache.spark.ml.feature.{OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object CriteoDataGenerator {

  private def buildCnt(
      sparkSession: SparkSession, df: DataFrame, rootPath: String): Unit = {
    df.createOrReplaceTempView("StringIndexedDF")
    for (i <- 0 until 26) {
      sparkSession.sql(s"select count(label)" +
        s" as cnt_per_category, category_index_$i from" +
        s" StringIndexedDF group by category_index_$i").
        createOrReplaceTempView(s"cnt_per_category_$i")
    }
    for (i <- 0 until 26) {
      sparkSession.sql(s"select count(category_index_$i) as pos_cnt_per_category," +
        s" category_index_$i from StringIndexedDF where label = 1.0 group by category_index_$i")
        .createOrReplaceTempView(s"pos_cnt_per_category_$i")
    }
    for (i <- 0 until 26) {
      sparkSession.sql(s"select pos_cnt_per_category * 1.0 / cnt_per_category as ratio_$i from" +
        s" cnt_per_category_$i, pos_cnt_per_category_$i" +
        s" where cnt_per_category_$i.category_index_$i ==" +
        s" pos_cnt_per_category_$i.category_index_$i").write.mode(
        SaveMode.Overwrite).save(rootPath + s"/ratio_category_$i")
    }
  }

  private def buildStringIndexingPipeline(): Pipeline = {
    val pipeline = new Pipeline()
    val stringIndexerArray = new Array[StringIndexer](26)
    for (i <- 0 until 26) {
      stringIndexerArray(i) = new StringIndexer().setInputCol(s"category_$i").setOutputCol(
        s"category_index_$i")
    }
    pipeline.setStages(stringIndexerArray)
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
    val stringIndexers = buildStringIndexingPipeline()
    val stringTransformedDF = stringIndexers.fit(typeTransformedDF).
      transform(typeTransformedDF)
    buildCnt(spark, stringTransformedDF, outputPath)
    // typeTransformedDF.write.format("parquet").mode(SaveMode.Overwrite).save(outputPath)
  }
}
