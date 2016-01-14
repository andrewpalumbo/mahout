/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.mahout.flinkbindings.drm

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.mahout.flinkbindings.{BlockifiedDrmDataSet, DrmDataSet, FlinkDistributedContext, wrapContext}
import org.apache.mahout.math.drm.{CheckpointedDrm, CacheHint, DrmLike}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.{DenseMatrix, Matrix, SparseRowMatrix}

import scala.math._
import scala.reflect._
import scala.util.Random

trait FlinkDrm[K] extends DrmLike[K] {
  def executionEnvironment: ExecutionEnvironment
  val context: FlinkDistributedContext
  def isBlockified: Boolean

  def asBlockified: BlockifiedFlinkDrm[K]
  def asRowWise: RowsFlinkDrm[K]

  //val keyClassTag: ClassTag[K] = implicitly[ClassTag[K]]
}

class RowsFlinkDrm[K: TypeInformation: ClassTag](val ds: DrmDataSet[K], val nCol: Int) extends FlinkDrm[K] {

  def executionEnvironment = ds.getExecutionEnvironment
  val context: FlinkDistributedContext = ds.getExecutionEnvironment

  def isBlockified = false
  
  override def canHaveMissingRows = true
  override def partitioningTag: Long = Random.nextLong()
  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = {
    this
  }

  /** R-like syntax for number of rows. */
  lazy val nrow: Long = ds.count()

  /** R-like syntax for number of columns */
  def ncol: Int = nCol


  def asBlockified : BlockifiedFlinkDrm[K] = {
    val ncolLocal = nCol
    val classTag = implicitly[ClassTag[K]]

    val parts = ds.mapPartition {
      values =>
        val (keys, vectors) = values.toIterable.unzip

        if (vectors.nonEmpty) {
          val vector = vectors.head
          val matrix: Matrix = if (vector.isDense) {
            val matrix = new DenseMatrix(vectors.size, ncolLocal)
            vectors.zipWithIndex.foreach { case (vec, idx) => matrix(idx, ::) := vec }
            matrix
          } else {
            new SparseRowMatrix(vectors.size, ncolLocal, vectors.toArray)
          }

          Seq((keys.toArray(classTag), matrix))
        } else {
          Seq()
        }
    }

    new BlockifiedFlinkDrm(parts, nCol)
  }

  def asRowWise = this

 val keyClassTag = implicitly[ClassTag[K]]
  if (keyClassTag == ClassTag.Any){
    throw new IllegalStateException("Illegal ClassTag Type: Any")
  }


}

class BlockifiedFlinkDrm[K: TypeInformation: ClassTag](val ds: BlockifiedDrmDataSet[K], val nCol: Int) extends FlinkDrm[K] {

  def executionEnvironment = ds.getExecutionEnvironment
  val context: FlinkDistributedContext = ds.getExecutionEnvironment

  override def canHaveMissingRows = true
  override def partitioningTag: Long = Random.nextLong()
  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = {
    this
  }

  /** R-like syntax for number of rows. */
  lazy val nrow: Long = ds.count()

  /** R-like syntax for number of columns */
  def ncol: Int = nCol



  def isBlockified = true

  def asBlockified = this

  def asRowWise = {
    val out = ds.flatMap {
      tuple =>
        val keys = tuple._1
        val block = tuple._2

        keys.view.zipWithIndex.map {
          case (key, idx) => (key, block(idx, ::))
        }
    }

    new RowsFlinkDrm(out, nCol)
  }

  val keyClassTag = implicitly[ClassTag[K]]
  if (keyClassTag == ClassTag.Any){
    throw new IllegalStateException("Illegal ClassTag Type: Any")
  }
}