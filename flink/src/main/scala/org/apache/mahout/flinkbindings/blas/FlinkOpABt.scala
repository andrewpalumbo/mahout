/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.flinkbindings.blas

import java.util

import org.apache.flink.api.scala._
import org.apache.flink.api.common.functions._
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala.DataSet
import org.apache.flink.util.Collector
import org.apache.mahout.logging._
import org.apache.mahout.math.drm.{BlockifiedDrmTuple, DrmTuple}
import org.apache.mahout.math.drm.logical.{OpAB, OpABt}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.{Matrix, SparseMatrix, SparseRowMatrix}
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm._
import org.apache.flink.configuration.Configuration
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.{BlockifiedFlinkDrm, FlinkDrm}
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.OpABt
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import org.apache.flink.api.scala.createTypeInformation

import scala.collection.JavaConversions

/** Contains DataSet plans for ABt operator */
object FlinkOpABt {

  private final implicit val log = getLog(FlinkOpABt.getClass)

  /**
   * General entry point for AB' operator.
   *
   * @param operator the AB' operator
   * @param srcA A source DataSet
   * @param srcB B source DataSet 
   * @tparam K
   */
  def abt[K: ClassTag: TypeInformation](
      operator: OpABt[K],
      srcA: FlinkDrm[K],
      srcB: FlinkDrm[Int]): FlinkDrm[K] = {


    debug("operator AB'(Flink)")
    abt_nograph[K](operator, srcA, srcB)
  }

  /**
   * Computes AB'
   *
   * General idea here is that we split both A and B vertically into blocks (one block per split),
   * then compute cartesian join of the blocks of both data sets. This creates tuples of the form of
   * (A-block, B-block). We enumerate A-blocks and transform this into (A-block-id, A-block, B-block)
   * and then compute A-block %*% B-block', thus producing tuples (A-block-id, AB'-block).
   *
   * The next step is to group the above tuples by A-block-id and stitch al AB'-blocks in the group
   * horizontally, forming single vertical block of the final product AB'.
   *
   * This logic is complicated a little by the fact that we have to keep block row and column keys
   * so that the stitching of AB'-blocks happens according to integer row indices of the B input.
   */
  private[flinkbindings] def abt_nograph[K: ClassTag: TypeInformation](
      operator: OpABt[K],
      srcA: FlinkDrm[K],
      srcB: FlinkDrm[Int]): FlinkDrm[K] = {

    // Blockify everything.
    val blocksA = srcA.asBlockified
    val blocksB = srcB.asBlockified

    val prodNCol = operator.ncol
    val prodNRow = operator.nrow
    println("A:"+ operator.A.collect)
    println("B:"+ operator.B.collect)

    // blockwise multiplication function
    def mmulFunc(tupleA: (Array[K], Matrix), tupleB: (Array[Int], Matrix)): (Array[K], Array[Int], Matrix) = {
      val (keysA, blockA) = tupleA
      val (keysB, blockB) = tupleB

      var ms = traceDo(System.currentTimeMillis())

      // We need to send keysB to the aggregator in order to know which columns are being updated.
      val result = (keysA, keysB, blockA %*% blockB.t)

      ms = traceDo(System.currentTimeMillis() - ms.get)
            trace(
              s"block multiplication of(${blockA.nrow}x${blockA.ncol} x ${blockB.ncol}x${blockB.nrow} is completed in $ms " +
                "ms.")
      trace(s"block multiplication types: blockA: ${blockA.getClass.getName}(${blockA.t.getClass.getName}); " +
              s"blockB: ${blockB.getClass.getName}.")

      result
    }


    implicit val typeInformation = createTypeInformation[(Array[K], Matrix)]
    implicit val typeInformation2 = createTypeInformation[(Int, (Array[K], Array[Int], Matrix))]
    implicit val typeInformation3 = createTypeInformation[(Array[K], Array[Int], Matrix)]

        val blockwiseMmulDataSet =

        // Combine blocks pairwise.
          pairwiseApply(blocksA.asBlockified.ds, blocksB.asBlockified.ds, mmulFunc)

            // Now reduce proper product blocks.
            // group by the partition key
            .groupBy(0)

            // Initalize the combiner as an empty transposed matrix:
            // (Op.A.ncol x partitionBlock.nrow).t
            // for each group (block partition)
            .combineGroup(new GroupCombineFunction[(Int, (Array[K], Array[Int], Matrix)),
                ((Array[K],  Matrix),(Array[K], Array[Int], Matrix))] {

               def combine(values: java.lang.Iterable[(Int, (Array[K], Array[Int], Matrix))],
                           out: Collector[((Array[K],  Matrix), (Array[K], Array[Int], Matrix))]): Unit = {
                   val tuple = values.iterator().next
                   val rowKeys = tuple._2._1
                   val colKeys = tuple._2._2
                   val block = tuple._2._3

                   // initialize the combiner as a sparse matrix.
                   // set each row of the transposed combiner to the column
                   // of the already block wise multiplied Matrix
                   val comb = new SparseMatrix(prodNCol, block.nrow).t
                   for ((col, i) <- colKeys.zipWithIndex) comb(::, col) := block(::, i)
                   out.collect((rowKeys, comb), (rowKeys, colKeys, block))
              }
            })


             // combine all members of each group into the above defined combiner
             .combineGroup(new GroupCombineFunction[((Array[K], Matrix),(Array[K], Array[Int], Matrix)),
                 (Array[K], Matrix)] {

                def combine(values: java.lang.Iterable[((Array[K], Matrix),(Array[K], Array[Int], Matrix))],
                  out: Collector[(Array[K], Matrix)]): Unit = {
                    val tuple = values.iterator().next()

                  // the combiner
                  val (rowKeys, c) = tuple._1

                  // the matrix
                  val (_, colKeys, block) = tuple._2
                  for ((col, i) <- colKeys.zipWithIndex) c(::, col) := block(::, i)

                  out.collect(rowKeys -> c)

                }
             })


            // now finally reduce the block
            .reduce( new ReduceFunction[(Array[K],Matrix)]{
                def reduce(mx1: (Array[K], Matrix), mx2: (Array[K], Matrix)): (Array[K], Matrix) = {

                   mx1._2 += mx2._2

                  println("/n/n/n"+mx1._2+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                   mx1._1 -> mx1._2
                }
            })



    val res = new BlockifiedFlinkDrm(ds = blockwiseMmulDataSet, ncol = prodNCol)
//    println(datasetWrap(res.asRowWise.ds).collect)
    res

  }
    /**
      * This function tries to use join instead of cartesian to group blocks together without bloating
      * the number of partitions. Hope is that we can apply pairwise reduction of block pair right away
      * so if the data to one of the join parts is streaming, the result is still fitting to memory,
      * since result size is much smaller than the operands.
      *
      * @param blocksA   blockified DataSet for A
      * @param blocksB   blockified DataSet for B
      * @param blockFunc a function over (blockA, blockB). Implies `blockA %*% blockB.t` but perhaps may be
      *                  switched to another scheme based on which of the sides, A or B, is bigger.
      */
      private def pairwiseApply[K1: ClassTag: TypeInformation, K2: ClassTag: TypeInformation,
                                 T: ClassTag: TypeInformation]
                                  ( blocksA: BlockifiedDrmDataSet[K1], blocksB: BlockifiedDrmDataSet[K2], blockFunc:
                                    ((Array[K1], Matrix), (Array[K2], Matrix)) =>
                            (Array[K1], Array[Int], Matrix) ): DataSet[(Int, (Array[K1], Array[Int], Matrix))] = {

      implicit val typeInformationA = createTypeInformation[(Int, Array[K1], Matrix)]
      implicit val typeInformationProd = createTypeInformation[(Int, (Array[K1], Array[Int], Matrix))]



      // calculate actual number of non empty partitions used by blocksA
      // we'll need this to key blocksB with the correct partition numbers
      // to join upon.  blocksA may use partitions 0,1 and blocksB may use partitions 2,3.
      val aNonEmptyParts = blocksA.map(new MapFunction[(Array[K1], Matrix), Int] {
        def map(a: (Array[K1], Matrix)): Int = {
          if (a._1.length > 0) {
            1
          } else {
            0
          }
        }
      }).reduce(new ReduceFunction[Int] {
        def reduce(a: Int, b: Int): Int = a + b
      }).collect().head

      // calculate actual number of non empty partitions used by blocksB
      // we'll need this to key blocksB with the correct partition numbers
      // to join upon.  blocksA may use partitions 0,1 and blocksB may use partitions 2,3.
      val bNonEmptyParts = blocksB.map(new MapFunction[(Array[K2], Matrix), Int] {
        def map(a: (Array[K2], Matrix)): Int = {
          if (a._1.length > 0) {
            1
          } else {
            0
          }
        }
      }).reduce(new ReduceFunction[Int] {
        def reduce(a: Int, b: Int): Int = a + b
      }).collect().head



      // get the first key of each Partition
      val blocksAFirstPartitionKey = blocksA.setParallelism(aNonEmptyParts).mapPartition( new RichMapPartitionFunction[(Array[K1], Matrix),
        (Int, K1) ] {
        // partition number
        var part: Int = 0

        // get the index of the partition
        override def open(params: Configuration): Unit = {
          part = getRuntimeContext.getIndexOfThisSubtask
        }

        // bind the partition number to each keySet/block so that we can put them back the correct order
        // presently this is only going to work for Int-Keyed Matrices.  String Keyed multiplications will
        // incorrect if partitioning is out of order
        def mapPartition(values: java.lang.Iterable[(Array[K1], Matrix)], out: Collector[(Int, K1)]): Unit  = {

          val blockIter = values.iterator()
          if (blockIter.hasNext()) {
            //map the task ID to the first key in the task
            val r = part -> blockIter.next._1(0)
            require(!blockIter.hasNext, s"more than 1 (${blockIter.asScala.size + 1}) blocks per partition and A of AB'")
            out.collect(r)
          }
        }
      })
        .collect()
        // reverse
        .map(x => x._2.asInstanceOf[Int] -> x._1)

        // sort on the value of the key
        .sortBy(_._1)

        // trow away the old partition task ids, they're no good.
        .unzip._1

        //rekey with 0 based partition identifiers to be used for the join
        .zipWithIndex

      // tuple to be broadcast to the mapPartition Function
      val partMap = blocksA.getExecutionEnvironment.fromCollection(blocksAFirstPartitionKey)


       // We will be joining blocks in B to blocks in A using A-partition indexes as a key.
       // we'll broadcast out key->partition identifier to this block
       // Prepare A side.
      val blocksAKeyed = blocksA.setParallelism(aNonEmptyParts).mapPartition( new RichMapPartitionFunction[(Array[K1], Matrix),
                                                            (Int, Array[K1], Matrix)] {
        // partition number
        var part: Int = 0

        var pMap: Map[Int, Int] = _
        // get the index of the partition
        override def open(params: Configuration): Unit = {
          val runtime = getRuntimeContext
          val dsX: util.List[(Int,Int)] = runtime.getBroadcastVariable("partMap")
          val parts: scala.collection.mutable.ArrayBuffer[(Int,Int)] = new scala.collection.mutable.ArrayBuffer[(Int,Int)]()
          val dsIter = dsX.asScala.toIterator
          parts.appendAll(dsIter)
           pMap = parts.map(x => x._1 -> x._2).toMap
         }

         // bind the partition number to each keySet/block
         def mapPartition(values: java.lang.Iterable[(Array[K1], Matrix)], out: Collector[(Int, Array[K1], Matrix)]): Unit  = {

           val blockIter = values.iterator()
           if (blockIter.hasNext()) {
             val keysBlock = blockIter.next
             // look up the correct order that this block should be in
             part = pMap(keysBlock._1.asInstanceOf[Array[Int]](0))
             val r = part -> (keysBlock._1 -> keysBlock._2)
             require(!blockIter.hasNext, s"more than 1 (${blockIter.asScala.size + 1}) blocks per partition and A of AB'")
             out.collect((r._1, r._2._1, r._2._2))
           }
         }
      }).withBroadcastSet(partMap, "partMap")

      println("\n\n\naNonEmptyPArts:"+aNonEmptyParts+"\n\n\n")
      println("\n\n\nbNonEmptyPArts:"+bNonEmptyParts+"\n\n\n")

      // throw away empty partitions
//      blocksAKeyed.setParallelism(aNonEmptyParts)

      // key the B blocks with the blocks of a assuming that they begin with 0 and are continuous
      // not sure if this assumption holds.

      implicit val typeInformationB = createTypeInformation[(Int, (Array[K2], Matrix))]


//      val blocksAKeyed =
//        blocksA.flatMap(new FlatMapFunction[(Array[K1], Matrix), (Int, Array[K1], Matrix)] {
//          var partsA = 0
//          def flatMap(in: (Array[K1], Matrix), out: Collector[(Int, Array[K1], Matrix)]): Unit = {
//            out.collect((partsA, in._1, in._2))
//            partsA += 1
//          }
//        })
//      blocksA.setParallelism(aNonEmptyParts)      // get the first key of each Partition
    val blocksBFirstPartitionKey = blocksB.setParallelism(aNonEmptyParts).mapPartition( new RichMapPartitionFunction[(Array[K2], Matrix),
      (Int, K2) ] {
      // partition number
      var part: Int = 0

      // get the index of the partition
      override def open(params: Configuration): Unit = {
        part = getRuntimeContext.getIndexOfThisSubtask
      }

      // bind the partition number to each keySet/block so that we can put them back the correct order
      // presently this is only going to work for Int-Keyed Matrices.  String Keyed multiplications will
      // incorrect if partitioning is out of order
      def mapPartition(values: java.lang.Iterable[(Array[K2], Matrix)], out: Collector[(Int, K2)]): Unit  = {

        val blockIter = values.iterator()
        if (blockIter.hasNext()) {
          //map the task ID to the first key in the task
          val r = part -> blockIter.next._1(0)
          require(!blockIter.hasNext, s"more than 1 (${blockIter.asScala.size + 1}) blocks per partition and A of AB'")
          out.collect(r)
        }
      }
    })
      .collect()
      // reverse
      .map(x => x._2.asInstanceOf[Int] -> x._1)

      // sort on the value of the key
      .sortBy(_._1)

      // trow away the old partition task ids, they're no good.
      .unzip._1

      //rekey with 0 based partition identifiers to be used for the join
      .zipWithIndex


      // tuple to be broadcast to the mapPartition Function
      val partBMap = blocksA.getExecutionEnvironment.fromCollection(blocksBFirstPartitionKey)

      val blocksBKeyed = blocksB.setParallelism(aNonEmptyParts).mapPartition( new RichMapPartitionFunction[(Array[K2], Matrix),
        (Int, Array[K2], Matrix)] {
        // partition number
        var part: Int = 0

        var pMap: Map[Int, Int] = _
        // get the index of the partition
        override def open(params: Configuration): Unit = {
          val runtime = getRuntimeContext
          val dsX: util.List[(Int,Int)] = runtime.getBroadcastVariable("partMap")
          val parts: scala.collection.mutable.ArrayBuffer[(Int,Int)] = new scala.collection.mutable.ArrayBuffer[(Int,Int)]()
          val dsIter = dsX.asScala.toIterator
          parts.appendAll(dsIter)
          pMap = parts.map(x => x._1 -> x._2).toMap
        }

        // bind the partition number to each keySet/block
        def mapPartition(values: java.lang.Iterable[(Array[K2], Matrix)], out: Collector[(Int, Array[K2], Matrix)]): Unit  = {

          val blockIter = values.iterator()
          if (blockIter.hasNext()) {
            val keysBlock = blockIter.next
            // look up the correct order that this block should be in
            if(!(keysBlock._1 == null)) {
              part = pMap.getOrElse(keysBlock._1.asInstanceOf[Array[Int]](0), 0)
              val r = part -> (keysBlock._1 -> keysBlock._2)
              require(!blockIter.hasNext, s"more than 1 (${blockIter.asScala.size + 1}) blocks per partition and A of AB'")
              out.collect((r._1, r._2._1, r._2._2))
            }
          }
        }
      }).withBroadcastSet(partBMap, "partMap")



//      val blocksBKeyed =
//        blocksB.setParallelism(aNonEmptyParts).flatMap(new RichMapFlatMapFunction[(Array[K2], Matrix), (Int, Array[K2], Matrix)] {
//
//          var pMap: Map[Int, Int] = _
//          var partsB = 0
//          // get the index of the partition
//          override def open(params: Configuration): Unit = {
//            val runtime = getRuntimeContext
//            val dsX: util.List[(Int, Int)] = runtime.getBroadcastVariable("partMap")
//            val parts: scala.collection.mutable.ArrayBuffer[(Int, Int)] = new scala.collection.mutable.ArrayBuffer[(Int, Int)]()
//            val dsIter = dsX.asScala.toIterator
//            parts.appendAll(dsIter)
//            pMap = parts.map(x => x._1 -> x._2).toMap
//          }
//
//
//          def mapPartition(in: (Array[K2], Matrix), out: Collector[(Int, Array[K2], Matrix)]): Unit = {
//            partsB = pMap(in._1.asInstanceOf[Array[Int]](0))
//            out.collect((partsB, in._1, in._2))
//
//          }
//        }).withBroadcastSet(partBMap, "partMap")


      println("\n\n\nBlocksA:")
      blocksAKeyed.collect().foreach{x => println(x._1 + " -> " + x._3)}
      println("--------------")
      println("\n\n\nBlocksB:")
      blocksBKeyed.collect().foreach{x => println(x._1 + " -> " + x._3)}
      println("Blocks after partition index mapping!!! \n\n\n")

      println("\n\n BlocksB.numPartitions:" + blocksBKeyed.collect().size)



      implicit val typeInformationJ = createTypeInformation[((Int,(Array[K1], Matrix), (Int, (Array[K2], Matrix))))]
      implicit val typeInformationJprod = createTypeInformation[(Int, T)]


      // Perform the inner join.
      val joined = blocksAKeyed.join(blocksBKeyed).where(0).equalTo(0)

        // Apply product function which should produce smaller products.
        // Hopefully, this streams blockB's in
      val mapped = joined.map { tuple => tuple._1._1 ->
          blockFunc(((tuple._1._2), (tuple._1._3)), (tuple._2._2, tuple._2._3))
        }

      println("\n\n\njoined---------")
      joined.collect().foreach{ x => println(x._1._1 + "->" + x._1._3)}
      joined.collect().foreach{ x => println(x._2._1 + "->" + x._2._3)}

      println("\n\n\nmapped---------")
      mapped.collect().foreach{ x => println(x._1+ "->" + x._2._3)}
      println("matrix---------")
     // mapped.collect().foreach{ x => println(x._1+ "->" + x._2._3)}

     // System.exit(1)
      mapped
      }

  }
