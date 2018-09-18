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

import scala.collection.mutable
import scala.io.Source

object LogProcessing {

  private def getSummaryTime(path: String): mutable.HashMap[Int, Long] = {
    val map = new mutable.HashMap[Int, Long]
    for (line <- Source.fromFile(path).getLines() if line.contains("cost (summary)")) {
      val array = line.split(" ")
      val indexOfTime = array.indexOf("rabit") - 1
      val time = array(indexOfTime).toLong
      val rabitRank = array(array.indexOf("rank:") + 1).toInt
      map += rabitRank -> time
    }
    map
  }

  private def getHistTime(path: String): mutable.HashMap[Int, Long] = {
    val map = new mutable.HashMap[Int, Long]
    for (line <- Source.fromFile(path).getLines() if line.contains("cost (hist)")) {
      val array = line.split(" ")
      val indexOfTime = array.indexOf("rabit") - 1
      val time = array(indexOfTime).toLong
      val rabitRank = array(array.indexOf("rank:") + 1).toInt
      map += rabitRank -> time
    }
    map
  }

  def main(args: Array[String]): Unit = {
    val path = args(0)
    val summaryMap = getSummaryTime(path)
    val histMap = getHistTime(path)
    summaryMap.toList.sortBy(_._1).foreach{ case (rabitRank, summaryTime) =>
      println(s"$rabitRank $summaryTime")}
    println("====================================")
    histMap.toList.sortBy(_._1).foreach{ case (rabitRank, histTime) =>
      println(s"$rabitRank $histTime")}
  }
}
