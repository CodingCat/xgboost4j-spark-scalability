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
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Row, SaveMode, SparkSession}

object CriteoDataGenerator {

  private def buildPipeline(): Pipeline = {
    val pipeline = new Pipeline()
    val stringIndexerArray = new Array[StringIndexer](26)
    for (i <- 0 until 26) {
      stringIndexerArray(i) = new StringIndexer().setInputCol(s"category-$i").setOutputCol(
        s"category-$i-index")
    }
    val numericColumns = (0 until 13).map(i => s"numeric-$i").toArray
    val categoryColumns = stringIndexerArray.map(_.getOutputCol)
    val vectorAssembler = new VectorAssembler().setInputCols(numericColumns ++ categoryColumns).
      setOutputCol("features")
    pipeline.setStages(stringIndexerArray :+ vectorAssembler)
  }

  def main(args: Array[String]): Unit = {
    val trainingInputPath = args(0)
    val outputPath = args(1)
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    val tsvFile = spark.read.format("csv").option("delimiter", "\t").load(trainingInputPath)
    val colNames = Seq("label") ++ (0 until 13).map(i => s"numeric-$i") ++
      (0 until 26).map(i => s"category-$i")
    val colRenamedDF = tsvFile.toDF(colNames: _*)
    val castExprArray = (0 until 13).map(i => s"cast (numeric-$i as double) numeric-$i")
    val typeTransformedDF = colRenamedDF.selectExpr(
      Seq("cast (label as double) label") ++ castExprArray: _*)
    val pipeline = buildPipeline()
    val transformedDF =
      pipeline.fit(typeTransformedDF).transform(typeTransformedDF).select("features", "label")
    transformedDF.write.format("parquet").mode(SaveMode.Overwrite).save(outputPath)
  }
}
