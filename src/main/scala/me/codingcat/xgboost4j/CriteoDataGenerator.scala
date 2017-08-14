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
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._

object CriteoDataGenerator {

  private def buildCnt(
      sparkSession: SparkSession, df: DataFrame, rootPath: String): DataFrame = {
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
    val maps = new Array[Map[String, Double]](26)
    for (i <- 0 until 26) {
      maps(i) =
        sparkSession.sql(s"select cnt_per_category_$i.category_index_$i as category_index_$i, " +
          s"pos_cnt_per_category * 1.0 / cnt_per_category as ratio_$i from" +
          s" cnt_per_category_$i, pos_cnt_per_category_$i" +
          s" where cnt_per_category_$i.category_index_$i ==" +
          s" pos_cnt_per_category_$i.category_index_$i").collect().map(row =>
          (row.getAs[String](s"category_index_$i"), row.getAs[Double](s"ratio_$i"))).toMap
    }
    sparkSession.catalog.dropTempView("StringIndexedDF")
    for (i <- 0 until 26) {
      sparkSession.catalog.dropTempView("cnt_per_category_$i")
      sparkSession.catalog.dropTempView("pos_cnt_per_category_$i")
    }
    var df1 = df
    for (i <- 0 until 26) {
      df1 = df1.withColumn(
        s"category_ratio_$i",
        udf{
          (category_index: String) => maps(i)(category_index)
        }.apply(col(s"category_index_$i")))
    }
    val droppedCols = for (i <- 0 until 26) yield s"category_index_$i"
    df1.drop(droppedCols: _*)
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
    val outputAsParquet = {
      if (args.length < 3) {
        false
      } else {
        args(3).toBoolean
      }
    }
    val spark = SparkSession.builder().getOrCreate()
    if (outputAsParquet) {
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
      stringTransformedDF.write.mode(SaveMode.Overwrite).parquet(outputPath)
    } else {
      val inputDF = spark.read.parquet(trainingInputPath)
      buildCnt(spark, inputDF, outputPath).write.format("parquet").mode(
        SaveMode.Overwrite).save(outputPath)
    }
  }
}
