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


import java.nio._
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
  //prepare 2D array from a 1-D array for a DenseMatrix constructor
  def array2dUnFlatten(array: Array[Double], numRows: Int, numCols: Int): Array[Array[Double]] = {

    val length: Int = array.length
    val mxValues = ofDim[Double](numRows,numCols)//(numCols)
    for (i <- 0 until numRows) {
      System.arraycopy(array, i * numCols, mxValues(i), 0, numCols)
    }
    mxValues
  }


  def makeDoubleBuffer(arr: Array[Double]): DoubleBuffer = {
    def bb: ByteBuffer = ByteBuffer.allocateDirect(arr.length * 8)
    bb.order(ByteOrder.nativeOrder())
    val db: DoubleBuffer = bb.asDoubleBuffer()
    db.put(arr)
    db.position(0)
    db
  }


  // Distributed Context to check for VCL Capabilities
  val vclCtx = new VclCtx()
  test("Simple dense %*% dense native mmul"){

    // probe to see if VCL libraries are installed
    if (vclCtx.useVCL) {

      val r1 = new Random(1234)

      val d = 300

      // Dense row-wise
      val mxA = new DenseMatrix(d, d)
      val mxB = new DenseMatrix(d, d)

      // add some data
      mxA := { (_,_,_) => r1.nextDouble()}
      mxB := { (_,_,_) => r1.nextDouble()}

      // time Mahout MMul
      // mxC = mxA %*% mxB via Mahout MMul
      val mxCControl = mxA %*% mxB


      //prepare 1-D array from mxA's backing set
      val flatMxA: Array[Double] = array2dFlatten(mxA.getBackingArray, mxA.nrow, mxA.ncol)
      val flatMxB: Array[Double] = array2dFlatten(mxB.getBackingArray, mxB.nrow, mxB.ncol)

      // mxC = mxA %*% mxB
      val flatMxC: Array[Double] = new Array[Double](mxA.nrow * mxB.ncol)

      // load the native library
      val vclblas: vcl_blas3 = new vcl_blas3()

      // mxC = mxA %*% mxB via VCL MMul
      dense_dense_mmul(new DoublePointer(DoubleBuffer.wrap(flatMxA)),
                       mxA.nrow.toLong, mxA.ncol.toLong,
                       new DoublePointer(DoubleBuffer.wrap(flatMxB)),
                       mxA.nrow.toLong, mxA.ncol.toLong,
                       new DoublePointer(DoubleBuffer.wrap(flatMxC))
      )

      val res = array2dUnFlatten(flatMxC, mxA.nrow, mxB.ncol)

      // shallow copy the resulting array into a DenseMatrix
      val mxCVCL = new DenseMatrix(res, true)


      mxCVCL.norm - mxCControl.norm should be < 1e-10

    } else {
      printf("No Native VCL library found... Skipping test")
    }
  }


  test("Simple dense %*% dense native MMul comparison with mahout MMul"){

    // probe to see if VCL libraries are installed
    if (vclCtx.useVCL) {

      val r1 = new Random(1234)

      val d1 = 500
      val d2 = 500
      val n = 3

      // Dense row-wise
      val mxA = new DenseMatrix(d1, d2)
      val mxB = new DenseMatrix(d2, d1)

      // add some data
       mxA := { (_,_,_) => r1.nextDouble()}
       mxB := { (_,_,_) => r1.nextDouble()}

      // time Mahout MMul
      val mahoutMsStart = System.currentTimeMillis()
      for(i <- 0 until n ) {
        // mxC = mxA %*% mxB via Mahout MMul
        val mxCControl = mxA %*% mxB
      }
      val mahoutTime = (System.currentTimeMillis() - mahoutMsStart).toDouble / n.toDouble


      val nativeMsStart = System.currentTimeMillis()
      for(i <- 0 until n ) {
        //prepare 1-D array from mxA's backing set
        val flatMxA: Array[Double] = array2dFlatten(mxA.getBackingArray, mxA.nrow, mxA.ncol)
        val flatMxB: Array[Double] = array2dFlatten(mxB.getBackingArray, mxB.nrow, mxB.ncol)

        // mxC = mxA %*% mxB
        val flatMxC: Array[Double] = new Array[Double](mxA.nrow * mxB.ncol)

        // load the native library
        val vclblas: vcl_blas3 = new vcl_blas3()

        // mxC = mxA %*% mxB via VCL MMul
        // using in-direct DoubleBuffers now need to compare speed with direct Buffers
        dense_dense_mmul(new DoublePointer(DoubleBuffer.wrap(flatMxA)),
          mxA.nrow.toLong, mxA.ncol.toLong,
          new DoublePointer(DoubleBuffer.wrap(flatMxB)),
          mxB.nrow.toLong, mxB.ncol.toLong,
          new DoublePointer(DoubleBuffer.wrap(flatMxC))
        )

        val res = array2dUnFlatten(flatMxC, mxA.nrow, mxB.ncol)
        val mxCVCL = new DenseMatrix(res, true)

      }
      val nativeTime = (System.currentTimeMillis() - nativeMsStart).toDouble / n.toDouble

      println("Mahout " + d1 + " x " + d2 + " dense mmul time: "+ mahoutTime + " ms ")
      println("Native " + d1 + " x " + d2 + " dense mmul time: "+ nativeTime + " ms ")
//      mxCVCL.norm - mxCControl.norm should be < 1e-6
    } else {
      printf("No Native VCL library found... Skipping test")
    }
  }
//  test("Simple dense %*% dense native MMul comparison with mahout MMul - using ByteBuffers"){
//
//    // probe to see if VCL libraries are installed
//    if (vclCtx.useVCL) {
//
//      val r1 = new Random(1234)
//
//      val d = 5000
//      val n = 3
//
//      // Dense row-wise
//      val mxA = new DenseMatrix(d, d)
//      val mxB = new DenseMatrix(d, d)
//
//      // add some data
//      mxA := { (_,_,_) => r1.nextDouble()}
//      mxB := { (_,_,_) => r1.nextDouble()}
//
//      // time Mahout MMul
//      val mahoutMsStart = System.currentTimeMillis()
//      for(i <- 0 until n ) {
//        // mxC = mxA %*% mxB via Mahout MMul
//        val mxCControl = mxA %*% mxB
//      }
//      val mahoutTime = (System.currentTimeMillis() - mahoutMsStart).toDouble / n.toDouble
//
//      val nativeMsStart = System.currentTimeMillis()
//      for(i <- 0 until n ) {
//        //prepare 1-D array from mxA's backing set
//        val flatMxA: Array[Double] = array2dFlatten(mxA.getBackingArray, mxA.nrow, mxA.ncol)
//        val flatMxB: Array[Double] = array2dFlatten(mxB.getBackingArray, mxB.nrow, mxB.ncol)
//
//        // mxC = mxA %*% mxB
//        val flatMxC: Array[Double] = new Array[Double](mxA.nrow * mxB.ncol)
//
//        // load the native library
//        val vclblas: vcl_blas3 = new vcl_blas3()
//
//        // mxC = mxA %*% mxB via VCL MMul
//        // using direct ByteBuffers now need to compare speed with in-direct DoubleBuffers
//        dense_dense_mmul(new DoublePointer(makeDoubleBuffer(flatMxA)),
//          mxA.nrow.toLong, mxA.ncol.toLong,
//          new DoublePointer(makeDoubleBuffer(flatMxB)),
//          mxA.nrow.toLong, mxA.ncol.toLong,
//          new DoublePointer(makeDoubleBuffer(flatMxC))
//        )
//
//        val res = array2dUnFlatten(flatMxC, mxA.nrow, mxB.ncol)
//        val mxCVCL = new DenseMatrix(res, true)
//
//      }
//      val nativeTime = (System.currentTimeMillis() - nativeMsStart).toDouble / n.toDouble
//
////      println("Mahout " + d + " x " + d + " dense mmul time: "+ mahoutTime + " ms ")
//      println("Native " + d + " x " + d + " dense mmul time: "+ nativeTime + " ms ")
//      //      mxCVCL.norm - mxCControl.norm should be < 1e-6
//    } else {
//      printf("No Native VCL library found... Skipping test")
//    }
//  }
//  test("Simple dense %*% dense native MMul comparison with mahout MMul using new DoublePointer(n)"){
//
//    // probe to see if VCL libraries are installed
//    if (vclCtx.useVCL) {
//
//      val r1 = new Random(1234)
//
//      val d = 50000
//      val n = 3
//
//      // Dense row-wise
//      val mxA = new DenseMatrix(d, d)
//      val mxB = new DenseMatrix(d, d)
//
//      // add some data
//      mxA := { (_,_,_) => r1.nextDouble()}
//      mxB := { (_,_,_) => r1.nextDouble()}
//
//      // time Mahout MMul
//      val mahoutMsStart = System.currentTimeMillis()
//      for(i <- 0 until n ) {
//        // mxC = mxA %*% mxB via Mahout MMul
//        val mxCControl = mxA %*% mxB
//      }
//      val mahoutTime = (System.currentTimeMillis() - mahoutMsStart).toDouble / n.toDouble
//
//
//      val nativeMsStart = System.currentTimeMillis()
//      for(i <- 0 until n ) {
//        //prepare 1-D array from mxA's backing set
//        val flatMxA: Array[Double] = array2dFlatten(mxA.getBackingArray, mxA.nrow, mxA.ncol)
//        val flatMxB: Array[Double] = array2dFlatten(mxB.getBackingArray, mxB.nrow, mxB.ncol)
//
//        // mxC = mxA %*% mxB
//        val flatMxC: Array[Double] = new Array[Double](mxA.nrow * mxB.ncol)
//
//        // load the native library
//        val vclblas: vcl_blas3 = new vcl_blas3()
//
//        // mxC = mxA %*% mxB via VCL MMul
//        // using in-direct DoubleBuffers now need to compare speed with direct Buffers
//        dense_dense_mmul(new DoublePointer(d*d).,
//          mxA.nrow.toLong, mxA.ncol.toLong,
//          new DoublePointer(DoubleBuffer.wrap(flatMxB)),
//          mxA.nrow.toLong, mxA.ncol.toLong,
//          new DoublePointer(DoubleBuffer.wrap(flatMxC))
//        )
//
//        val res = array2dUnFlatten(flatMxC, mxA.nrow, mxB.ncol)
//        val mxCVCL = new DenseMatrix(res, true)
//
//      }
//      val nativeTime = (System.currentTimeMillis() - nativeMsStart).toDouble / n.toDouble
//
//      println("Mahout " + d + " x " + d + " dense mmul time: "+ mahoutTime + " ms ")
//      println("Native " + d + " x " + d + " dense mmul time: "+ nativeTime + " ms ")
//      //      mxCVCL.norm - mxCControl.norm should be < 1e-6
//    } else {
//      printf("No Native VCL library found... Skipping test")
//    }
//  }
}

