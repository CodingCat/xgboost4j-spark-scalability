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

import org.apache.spark.sql.DataFrame

object AirlineDataGenerator {

  private val rawInputDFList = new ListBuffer[DataFrame]

  def main(args: Array[String]): Unit = {
    import scala.collection.JavaConverters._
    val config = ConfigFactory.parseFile(new File(args(0)))
    // 1. compile the list of input files (different years)
    val inputFileList = config.getStringList(
      "me.codingcat.xgboost4j.dataset.airline.paths")
    for (airlineFilePath <- inputFileList.asScala) {
        
    }
    // 2. generate DataFrame of the all airline data
    // 3. extract columns
    // 4. sample from dataframe and save to output path
  }
}
