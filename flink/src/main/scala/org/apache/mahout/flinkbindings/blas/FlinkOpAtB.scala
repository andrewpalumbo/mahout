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
package org.apache.mahout.flinkbindings.blas

import java.lang.Iterable

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.math.BigInt.int2bigInt
import scala.reflect.ClassTag

import org.apache.flink.api.common.functions.{ReduceFunction, MapFunction, FlatMapFunction, GroupReduceFunction}
import org.apache.flink.api.scala.DataSet
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.OpAtB
import org.apache.mahout.math.scalabindings.RLikeOps._

import com.google.common.collect.Lists

import org.apache.flink.api.scala._

/**
 * Implementation is taken from Spark's AtB
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/AtB.scala
 */
object FlinkOpAtB {

  def notZippable[A](operator: OpAtB[A], At: FlinkDrm[A], B: FlinkDrm[A]): FlinkDrm[Int] = {

//    val rowsAt = At.asRowWise.ds.asInstanceOf[DrmDataSet[A]]
//    val rowsB = B.asRowWise.ds.asInstanceOf[DrmDataSet[A]]
//    val joined = rowsAt.join(rowsB).where(0).equalTo(0)
//
//    val ncol = op.ncol
//    val nrow = op.nrow.toInt
//    val blockHeight = 10
//    val blockCount = safeToNonNegInt((nrow - 1) / blockHeight + 1)


    val prodNCol = operator.ncol
    val prodNRow = safeToNonNegInt(operator.nrow)
    val aNRow = safeToNonNegInt(operator.A.nrow)

    val rowsAt = At.asRowWise.ds.asInstanceOf[DrmDataSet[A]]
    val rowsB = B.asRowWise.ds.asInstanceOf[DrmDataSet[A]]
//    val joined = rowsAt.join(rowsB).where(0).equalTo(0)
    val joined = rowsAt.join(rowsB).where(0).equalTo(0)

    val partitionsAt: DataSet[Int] =
      At.asBlockified.ds.map(new MapFunction[(Array[A], Matrix), Int] {
      def map(a: (Array[A], Matrix)): Int = 1
    }).reduce(new ReduceFunction[Int] {
      def reduce(a: Int, b: Int): Int = a + b
    })

    val partitionsB: DataSet[Int] =
      B.asBlockified.ds.map(new MapFunction[(Array[A], Matrix), Int] {
        def map(a: (Array[A], Matrix)): Int = 1
      }).reduce(new ReduceFunction[Int] {
        def reduce(a: Int, b: Int): Int = a + b
      })

//    val partitionsJ: DataSet[Int] =
//      joined.mapPartition(new MapFunction[(Array[A], Matrix), Int] {
//        def map(a: (Array[A], Matrix)): Int = 1
//      }).reduce(new ReduceFunction[Int] {
//        def reduce(a: Int, b: Int): Int = a + b
//      })


    val aPartitions = partitionsAt.collect()(0)
    val bPartitions = partitionsB.collect()(0)

    // Approximate number of final partitions. We take bigger partitions as our guide to number of
    // elements per partition. TODO: do it better.
    // Elements per partition, bigger of two operands.
    val epp = aNRow.toDouble * prodNRow / aPartitions max aNRow.toDouble * prodNCol /
      bPartitions

    // Number of partitions we want to converge to in the product. For now we simply extrapolate that
    // assuming product density and operand densities being about the same; and using the same element
    // per partition number in the product as the bigger of two operands.
    val numProductPartitions = (prodNCol.toDouble * prodNRow / epp).ceil.toInt //min prodNRow

   // val blockHeight = (epp * (bPartitions max aPartitions)).ceil.toInt
    // Figure out appriximately block height per partition of the result.
    val blockHeight = safeToNonNegInt((aNRow - 1) / (bPartitions max aPartitions)) + 1
    //val numProductPartitions = (prodNCol.toDouble * prodNRow / epp).ceil.toInt




    val preProduct: DataSet[(Int, Matrix)] =
             joined.flatMap(new FlatMapFunction[((A, Vector), (A, Vector)), (Int, Matrix)] {
      def flatMap(in: ((A, Vector), (A, Vector)), out: Collector[(Int, Matrix)]): Unit = {
        val avec = in._1._2
        val bvec = in._2._2


//        0.until((bPartitions max  aPartitions)) map { blockKey =>
          0.until((numProductPartitions)) map { blockKey =>
          val blockStart = blockKey * blockHeight
          val blockEnd = prodNCol min (blockStart + blockHeight)


           // val outer = avec(blockStart.ceil.toInt until (blockEnd.ceil.toInt min avec.size-1)) cross bvec
          val outer = avec(blockStart until blockEnd) cross bvec
         // val outer = avec cross bvec
          out.collect(blockKey -> outer)
          out
        }
      }
    })

    val res: BlockifiedDrmDataSet[Int] = 
      preProduct.groupBy(0).reduceGroup(new GroupReduceFunction[(Int, Matrix), BlockifiedDrmTuple[Int]] {
      def reduce(values: Iterable[(Int, Matrix)], out: Collector[BlockifiedDrmTuple[Int]]): Unit = {
        val it = Lists.newArrayList(values).asScala
        val (idx, _) = it.head

        val block = it.map { t => t._2 }.reduce { (m1, m2) => m1 + m2 }

       // val blockKey = idx.until(block.nrow).toArray[Int]

        val blockStart = idx * blockHeight
        val keys = Array.tabulate(block.nrow)(blockStart + _)
        out.collect(keys -> block)
      }

    }).setParallelism(bPartitions max aPartitions).rebalance()
//        // Throw away block key, generate row keys instead.
//        .map({ tuple =>
//          val blockKey = tuple._1.head
//          val block = tuple._2
//
//          val blockStart = blockKey * blockHeight
//          val rowKeys = Array.tabulate(block.nrow)(blockStart + _)
//          (rowKeys -> block)
//      })


    //throw away the keys
    new BlockifiedFlinkDrm[Int](res, prodNCol)
  }

}