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

package me.codingcat.xgboost4j.common

import scala.collection.mutable

import com.typesafe.config.Config

private[xgboost4j] object Utils {

  private val params = Map(
    "max_depth" -> 6,
    "min_child_weight" -> 1,
    "gamma" -> 0,
    "subsample" -> 1,
    "colsample_bytree" -> 1,
    "scale_pos_weight" -> 1,
    "silent" -> 0,
    "eta" -> 0.3,
    "objective" -> "binary:logistic"
  )

  def fromConfigToXGBParams(config: Config): Map[String, Any] = {
    val specifiedMap = new mutable.HashMap[String, Any]
    for (name <- params.keys) {
      if (config.hasPath(s"me.codingcat.xgboost4j.$name")) {
        specifiedMap += name -> config.getAnyRef(s"me.codingcat.xgboost4j.$name").asInstanceOf[Any]
      }
    }
    params ++ specifiedMap.toMap
  }
}
