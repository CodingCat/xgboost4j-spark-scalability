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

package org.apache.spark.ml.feature

import org.apache.spark.annotation.Since
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DoubleType, NumericType, StructType}

/**
 * A one-hot encoder that maps a column of category indices to a column of binary vectors, with
 * at most a single one-value per row that indicates the input category index.
 * For example with 5 categories, an input value of 2.0 would map to an output vector of
 * `[0.0, 0.0, 1.0, 0.0]`.
 * The last category is not included by default (configurable via `OneHotEncoder!.dropLast`
 * because it makes the vector entries sum up to one, and hence linearly dependent.
 * So an input value of 4.0 maps to `[0.0, 0.0, 0.0, 0.0]`.
 *
 * @note This is different from scikit-learn's OneHotEncoder, which keeps all categories.
 * The output vectors are sparse.
 *
 * @see `StringIndexer` for converting categorical values into category indices
 */
@Since("1.4.0")
class OneHotEncoder2 @Since("1.4.0") (@Since("1.4.0") override val uid: String) extends Transformer
  with HasInputCol with HasOutputCol with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("oneHot"))

  /**
   * Whether to drop the last category in the encoded vector (default: true)
   * @group param
   */
  @Since("1.4.0")
  final val dropLast: BooleanParam =
  new BooleanParam(this, "dropLast", "whether to drop the last category")
  setDefault(dropLast -> true)

  final val featureSize: IntParam = new IntParam(this, "featureSize", "the size of features")
  setDefault(featureSize -> 0)

  def setFeatureSize(fSize: Int): Unit = {
    set(featureSize, fSize)
  }

  /** @group getParam */
  @Since("2.0.0")
  def getDropLast: Boolean = $(dropLast)

  /** @group setParam */
  @Since("1.4.0")
  def setDropLast(value: Boolean): this.type = set(dropLast, value)

  /** @group setParam */
  @Since("1.4.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    require(schema(inputColName).dataType.isInstanceOf[NumericType],
      s"Input column must be of type NumericType but got ${schema(inputColName).dataType}")
    val inputFields = schema.fields
    require(!inputFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")

    val inputAttr = Attribute.fromStructField(schema(inputColName))
    val outputAttrNames: Option[Array[String]] = inputAttr match {
      case nominal: NominalAttribute =>
        if (nominal.values.isDefined) {
          nominal.values
        } else if (nominal.numValues.isDefined) {
          nominal.numValues.map(n => Array.tabulate(n)(_.toString))
        } else {
          None
        }
      case binary: BinaryAttribute =>
        if (binary.values.isDefined) {
          binary.values
        } else {
          Some(Array.tabulate(2)(_.toString))
        }
      case _: NumericAttribute =>
        throw new RuntimeException(
          s"The input column $inputColName cannot be numeric.")
      case _ =>
        None // optimistic about unknown attributes
    }

    val filteredOutputAttrNames = outputAttrNames.map { names =>
      if ($(dropLast)) {
        require(names.length > 1,
          s"The input column $inputColName should have at least two distinct values.")
        names.dropRight(1)
      } else {
        names
      }
    }

    val outputAttrGroup = if (filteredOutputAttrNames.isDefined) {
      val attrs: Array[Attribute] = filteredOutputAttrNames.get.map { name =>
        BinaryAttribute.defaultAttr.withName(name)
      }
      new AttributeGroup($(outputCol), attrs)
    } else {
      new AttributeGroup($(outputCol))
    }

    val outputFields = inputFields :+ outputAttrGroup.toStructField()
    StructType(outputFields)
  }

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    // schema transformation
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)
    val shouldDropLast = $(dropLast)
    var outputAttrGroup = AttributeGroup.fromStructField(
      transformSchema(dataset.schema)(outputColName))
    if (outputAttrGroup.size < 0) {
      // If the number of attributes is unknown, we check the values from the input column.
      val numAttrs = dataset.select(col(inputColName).cast(DoubleType)).rdd.map(_.getDouble(0))
        .aggregate(0.0)(
          (m, x) => {
            assert(x <= Int.MaxValue,
              s"OneHotEncoder only supports up to ${Int.MaxValue} indices, but got $x")
            assert(x >= 0.0 && x == x.toInt,
              s"Values from column $inputColName must be indices, but got $x.")
            math.max(m, x)
          },
          (m0, m1) => {
            math.max(m0, m1)
          }
        ).toInt + 1
      val outputAttrNames = Array.tabulate(numAttrs)(_.toString)
      val filtered = if (shouldDropLast) outputAttrNames.dropRight(1) else outputAttrNames
      val outputAttrs: Array[Attribute] =
        filtered.map(name => BinaryAttribute.defaultAttr.withName(name))
      outputAttrGroup = new AttributeGroup(outputColName, outputAttrs)
    }
    val metadata = outputAttrGroup.toMetadata()

    // data transformation
    val size = outputAttrGroup.size
    val oneValue = Array(1.0)
    val emptyValues = Array.empty[Double]
    val emptyIndices = Array.empty[Int]
    val encode = udf { label: Double =>
      if (label < size) {
        Vectors.sparse(size, Array(label.toInt), oneValue)
      } else {
        Vectors.sparse(size, emptyIndices, emptyValues)
      }
    }

    dataset.select(col("*"), encode(col(inputColName).cast(DoubleType)).as(outputColName, metadata))
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): OneHotEncoder = defaultCopy(extra)
}

@Since("1.6.0")
object OneHotEncoder extends DefaultParamsReadable[OneHotEncoder] {

  @Since("1.6.0")
  override def load(path: String): OneHotEncoder = super.load(path)
}
