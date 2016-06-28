package org.apache.mahout.viennacl
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import java.nio.{ByteBuffer, DoubleBuffer}
import java.util.Random

import org.apache.mahout.javacpp.linalg.vcl_blas3._
import org.bytedeco.javacpp._
import org.scalatest.{FunSuite, Matchers}
import org.apache.mahout.math._
import drm._
import scalabindings._
import RLikeOps._
import org.apache.mahout.javacpp.linalg.vcl_blas3

import scala.collection.JavaConversions
import Array._

class HelloVCLTestSuite extends FunSuite with Matchers {

  class VclCtx extends DistributedContext{
    val engine: DistributedEngine = null

    def close() {
    }
  }

  //prepare 1-D array from a DenseMatrix backing set
  def array2dFlatten(array: Array[Array[Double]], numRows: Int, numCols: Int): Array[Double] = {

    var pos: Int = 0
    val mxValues = new Array[Double](numRows * numCols: Int)
    for (ar: Array[Double] <- array) {
      System.arraycopy(ar, 0, mxValues, pos, ar.length)
      pos = pos + ar.length
    }
    mxValues
  }
  //prepare 1-D array from a DenseMatrix backing set
  def array2dUnFlatten(array: Array[Double], numRows: Int, numCols: Int): Array[Array[Double]] = {

    val length: Int = array.length
    val mxValues = ofDim[Double](numRows,numCols)//(numCols)
    for (i <- 0 until numRows) {
      System.arraycopy(array, i * numCols, mxValues(i), 0, numCols)

    }
    mxValues
  }


  // Distributed Context to check for VCL Capabilities
  val vclCtx = new VclCtx()





  test("Simple dense %*% dense native mmul"){

    // probe to see if VCL libraries are installed
    if (vclCtx.useVCL) {

      val r1 = new Random(1234)

      val d = 1000
      val n = 3

      // Dense row-wise
      val mxA = new DenseMatrix(d, d) //:= { (_,_,_) => r1.nextDouble()}
      val mxB = new DenseMatrix(d, d) //:= { (_,_,_) => r1.nextDouble()}

      // add some data
       mxA := { (_,_,_) => r1.nextDouble()}
       mxB := { (_,_,_) => r1.nextDouble()}

     // val mxC = dense(1000,1000)

      // mxC = mxA %*% mxB via Mahout MMul
      val mxCControl = mxA %*% mxB


      //prepare 1-D array from mxA's backing set
      val flatMxA: Array[Double] = array2dFlatten(mxA.getBackingArray, mxA.nrow, mxA.ncol)
      val flatMxB: Array[Double]  = array2dFlatten(mxB.getBackingArray, mxB.nrow, mxB.ncol)

      // mxC = mxA %*% mxB
      val flatMxC: Array[Double] = new Array[Double](mxA.nrow * mxB.ncol)

      val vclblas: vcl_blas3 = new vcl_blas3()

      // mxC = mxA %*% mxB via VCL MMul
      dense_dense_mmul(flatMxA,
                       mxA.nrow.toLong, mxA.ncol.toLong,
                       flatMxB,
                       mxA.nrow.toLong, mxA.ncol.toLong,
                       flatMxC
      )

      val res = array2dUnFlatten(flatMxC,mxA.nrow,mxB.ncol)
      val mxCVCL = new DenseMatrix(res, true)

      mxCVCL.norm - mxCControl.norm should be < 1e-6



    } else {
      printf("No Native VCL library found... Skipping test")
    }
  }
}

