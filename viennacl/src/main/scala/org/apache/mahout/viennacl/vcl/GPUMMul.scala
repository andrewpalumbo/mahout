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

package org.apache.mahout.viennacl.vcl

import org.apache.mahout.logging._
import org.apache.mahout.math
import org.apache.mahout.math._
import org.apache.mahout.math.flavor.{BackEnum, TraversingStructureEnum}
import org.apache.mahout.math.function.Functions
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.viennacl.vcl.javacpp.Functions._
import org.apache.mahout.viennacl.vcl.javacpp.LinalgFunctions._
import org.apache.mahout.viennacl.vcl.javacpp.{CompressedMatrix, Context, DenseRowMatrix}

import scala.collection.JavaConversions._

object GPUMMul extends MMBinaryFunc {

  private final implicit val log = getLog(GPUMMul.getClass)

  override def apply(a: Matrix, b: Matrix, r: Option[Matrix]): Matrix = {

    require(a.ncol == b.nrow, "Incompatible matrix sizes in matrix multiplication.")

    val (af, bf) = (a.getFlavor, b.getFlavor)
    val backs = (af.getBacking, bf.getBacking)
    val sd = (af.getStructure, math.scalabindings.densityAnalysis(a), bf.getStructure, densityAnalysis(b))

    debug("\n\n a nonzeros: "+a.getNumNondefaultElements)
    debug("\n b nonzeros: "+b.getNumNondefaultElements+"\n\n")

    val alg: MMulAlg = backs match {

      // Both operands are jvm memory backs.
      case (BackEnum.JVMMEM, BackEnum.JVMMEM) ⇒

        sd match {

          // Multiplication cases by a diagonal matrix.
          case (TraversingStructureEnum.VECTORBACKED, _, TraversingStructureEnum.COLWISE, _)
            if a.isInstanceOf[DiagonalMatrix] ⇒ jvmDiagCW
          case (TraversingStructureEnum.VECTORBACKED, _, TraversingStructureEnum.SPARSECOLWISE, _)
            if a.isInstanceOf[DiagonalMatrix] ⇒ jvmDiagCW
          case (TraversingStructureEnum.VECTORBACKED, _, TraversingStructureEnum.ROWWISE, _)
            if a.isInstanceOf[DiagonalMatrix] ⇒ jvmDiagRW
          case (TraversingStructureEnum.VECTORBACKED, _, TraversingStructureEnum.SPARSEROWWISE, _)
            if a.isInstanceOf[DiagonalMatrix] ⇒ jvmDiagRW

          case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.VECTORBACKED, _)
            if b.isInstanceOf[DiagonalMatrix] ⇒ jvmCWDiag
          case (TraversingStructureEnum.SPARSECOLWISE, _, TraversingStructureEnum.VECTORBACKED, _)
            if b.isInstanceOf[DiagonalMatrix] ⇒ jvmCWDiag
          case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.VECTORBACKED, _)
            if b.isInstanceOf[DiagonalMatrix] ⇒ jvmRWDiag
          case (TraversingStructureEnum.SPARSEROWWISE, _, TraversingStructureEnum.VECTORBACKED, _)
            if b.isInstanceOf[DiagonalMatrix] ⇒ jvmRWDiag

          // Dense-dense cases
          case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) if a eq b.t ⇒ gpuDRWAAt
          case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) if a.t eq b ⇒ gpuDRWAAt
          case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) ⇒ jvmRWCW
          case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.ROWWISE, true) ⇒ jvmRWRW
          case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.COLWISE, true) ⇒ jvmCWCW
          case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) if a eq b.t ⇒ jvmDCWAAt
          case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) if a.t eq b ⇒ jvmDCWAAt
          case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) ⇒ jvmCWRW

          // Sparse row matrix x sparse row matrix (array of vectors)
          case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.ROWWISE, false) ⇒ gpuSparseRWRW
          case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.COLWISE, false) ⇒ jvmSparseRWCW
          case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.ROWWISE, false) ⇒ jvmSparseCWRW
          case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.COLWISE, false) ⇒ jvmSparseCWCW

          // Sparse matrix x sparse matrix (hashtable of vectors)
          case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.SPARSEROWWISE, false) ⇒
            gpuSparseRowRWRW
          case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.SPARSECOLWISE, false) ⇒
            jvmSparseRowRWCW
          case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.SPARSEROWWISE, false) ⇒
            jvmSparseRowCWRW
          case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.SPARSECOLWISE, false) ⇒
            jvmSparseRowCWCW

          // Sparse matrix x non-like
          case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ gpuSparseRowRWRW
          case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseRowRWCW
          case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseRowCWRW
          case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseCWCW
          case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.SPARSEROWWISE, false) ⇒ gpuSparseRWRW
          case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.SPARSECOLWISE, false) ⇒ jvmSparseRWCW
          case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.SPARSEROWWISE, false) ⇒ jvmSparseCWRW
          case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.SPARSECOLWISE, false) ⇒ jvmSparseRowCWCW

          // Everything else including at least one sparse LHS or RHS argument
          case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ gpuSparseRWRW
          case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseRWCW
          case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseCWRW
          case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseCWCW2flips

          // Sparse methods are only effective if the first argument is sparse, so we need to do a swap.
          case (_, _, _, false) ⇒ (a, b, r) ⇒ apply(b.t, a.t, r.map {_.t}).t

          // Default jvm-jvm case.
          case _ ⇒ jvmRWCW
        }
    }

    alg(a, b, r)
  }

  type MMulAlg = MMBinaryFunc

  @inline
  private def jvmRWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {

    require(r.forall(mxR ⇒ mxR.nrow == a.nrow && mxR.ncol == b.ncol))
    val (m, n) = (a.nrow, b.ncol)

    val mxR = r.getOrElse(if (densityAnalysis(a)) a.like(m, n) else b.like(m, n))

    for (row ← 0 until mxR.nrow; col ← 0 until mxR.ncol) {
      // this vector-vector should be sort of optimized, right?
      mxR(row, col) = a(row, ::) dot b(::, col)
    }
    mxR
  }


  @inline
  private def jvmRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {

    // A bit hackish: currently, this relies a bit on the fact that like produces RW(?)
    val bclone = b.like(b.ncol, b.nrow).t
    for (brow ← b) bclone(brow.index(), ::) := brow

    require(bclone.getFlavor.getStructure == TraversingStructureEnum.COLWISE || bclone.getFlavor.getStructure ==
      TraversingStructureEnum.SPARSECOLWISE, "COL wise conversion assumption of RHS is wrong, do over this code.")

    jvmRWCW(a, bclone, r)
  }

  private def jvmCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    jvmRWRW(b.t, a.t, r.map(_.t)).t
  }

  private def jvmCWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    // This is a primary contender with Outer Prod sum algo.
    // Here, we force-reorient both matrices and run RWCW.
    // A bit hackish: currently, this relies a bit on the fact that clone always produces RW(?)
    val aclone = a.cloned

    require(aclone.getFlavor.getStructure == TraversingStructureEnum.ROWWISE || aclone.getFlavor.getStructure ==
      TraversingStructureEnum.SPARSEROWWISE, "Row wise conversion assumption of RHS is wrong, do over this code.")

    jvmRWRW(aclone, b, r)
  }

  // left is Sparse right is any
  private def gpuSparseRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    val mxR = r.getOrElse(b.like(a.nrow, b.ncol))

//    // This is basically almost the algorithm from SparseMatrix.times
//    for (arow ← a; ael ← arow.nonZeroes)
//      mxR(arow.index(), ::).assign(b(ael.index, ::), Functions.plusMult(ael))
//
//    mxR

    // make sure that the matrix is not empty.  VCL {{compressed_matrix}}s must
    // hav nnz > 0
    val hasElementsA = a.getNumNondefaultElements.sum - a.numRows() * a.numCols() == 0
    val hasElementsB = b.getNumNondefaultElements.sum - b.numRows() * b.numCols() == 0

    // A has a sparse matrix structure of unknown size.  We do not want to
    // simply convert it to a Dense Matrix which may result in an OOM error.
    // If it is empty use JVM MMul, since we can not convert it to a VCL CSR Matrix.
    if (!hasElementsA)  {
     return MMul(a, b, r)
    }

    // CSR matrices are efficient up to 50% non-zero
    if(b.getFlavor.isDense) {
      var ms = System.currentTimeMillis()
      val oclCtx = new Context(Context.OPENCL_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, oclCtx)
      val oclB = toVclDenseRM(b, oclCtx)
      val oclC = new DenseRowMatrix(prod(oclA, oclB))
      val mxC = fromVclDenseRM(oclC)
      ms = System.currentTimeMillis() - ms
      debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()

      mxC
    } else {
      // Fall back to JVM based MMul if either matrix is sparse and empty
      if ((!(hasElementsA)) || ((!(hasElementsB))))  {
        return MMul(a, b, r)
      }

      var ms = System.currentTimeMillis()
      val oclCtx = new Context(Context.OPENCL_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, oclCtx)
      val oclB = toVclCmpMatrixAlt(b, oclCtx)
      val oclC = new CompressedMatrix(prod(oclA, oclB))
      val mxC = fromVclCompressedMatrix(oclC)
      ms = System.currentTimeMillis() - ms
      debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()

      mxC
    }

  }

  //sparse %*% dense
  private def gpuSparseRowRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {

    val hasElementsA = a.getNumNondefaultElements.sum - a.numRows() * a.numCols() == 0

    // A has a sparse matrix structure of unknown size.  We do not want to
    // simply convert it to a Dense Matrix which may result in an OOM error.
    // If it is empty fall back to  JVM MMul, since we can not convert it
    // to a VCL CSR Matrix.
    if (!hasElementsA)  {
      return MMul(a, b, r)
    }

    var ms = System.currentTimeMillis()
    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val oclA = toVclCmpMatrixAlt(a, oclCtx)
    val oclB = toVclDenseRM(b, oclCtx)
    val oclC = new DenseRowMatrix(prod(oclA, oclB))
    val mxC = fromVclDenseRM(oclC)
    ms = System.currentTimeMillis() - ms
    debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

    oclA.close()
    oclB.close()
    oclC.close()

    mxC
  }

  private def jvmSparseRowCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRowRWRW(b.t, a.t, r.map(_.t)).t

  private def jvmSparseRowCWCW2flips(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRowRWRW(a cloned, b cloned, r)

  private def jvmSparseRowRWCW(a: Matrix, b: Matrix, r: Option[Matrix]) =
    gpuSparseRowRWRW(a, b cloned, r)


  private def jvmSparseRowCWRW(a: Matrix, b: Matrix, r: Option[Matrix]) =
    gpuSparseRowRWRW(a cloned, b, r)

  private def jvmSparseRWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRWRW(a, b.cloned, r)

  private def jvmSparseCWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRWRW(a cloned, b, r)

  private def jvmSparseCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRWRW(b.t, a.t, r.map(_.t)).t

  private def jvmSparseCWCW2flips(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRWRW(a cloned, b cloned, r)

  private def jvmDiagRW(diagm:Matrix, b:Matrix, r:Option[Matrix] = None):Matrix = {
    val mxR = r.getOrElse(b.like(diagm.nrow, b.ncol))

    for (del ← diagm.diagv.nonZeroes())
      mxR(del.index, ::).assign(b(del.index, ::), Functions.plusMult(del))

    mxR
  }

  private def jvmDiagCW(diagm: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    val mxR = r.getOrElse(b.like(diagm.nrow, b.ncol))
    for (bcol ← b.t) mxR(::, bcol.index()) := bcol * diagm.diagv
    mxR
  }

  private def jvmCWDiag(a: Matrix, diagm: Matrix, r: Option[Matrix] = None) =
    jvmDiagRW(diagm, a.t, r.map {_.t}).t

  private def jvmRWDiag(a: Matrix, diagm: Matrix, r: Option[Matrix] = None) =
    jvmDiagCW(diagm, a.t, r.map {_.t}).t


  /** Dense column-wise AA' */
  private def jvmDCWAAt(a:Matrix, b:Matrix, r:Option[Matrix] = None) = {
    // a.t must be equiv. to b. Cloning must rewrite to row-wise.
    gpuDRWAAt(a.cloned,null,r)
  }

  /** Dense Row-wise AA' */
  // we probably will not want to use this for the actual release unless A is cached already
  // but adding for testing purposes.
  private def gpuDRWAAt(a:Matrix, b:Matrix, r:Option[Matrix] = None) = {
    // a.t must be equiv to b.

    debug("AAt computation detected; passing off to GPU")

    // Check dimensions if result is supplied.
    require(r.forall(mxR ⇒ mxR.nrow == a.nrow && mxR.ncol == a.nrow))

    val mxR = r.getOrElse(a.like(a.nrow, a.nrow))

    var ms = System.currentTimeMillis()
    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val oclA = toVclDenseRM(src = a, oclCtx)
    // TODO: BADHACK A' getting memory errors using A twice
    // val oclApr = toVclDenseRM(src = a cloned, oclCtx)
    val oclAt = new DenseRowMatrix(trans(oclA))
    val oclC = new DenseRowMatrix(prod(oclA, oclAt))

    val mxC = fromVclDenseRM(oclC)
    ms = System.currentTimeMillis() - ms
    debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

    oclA.close()
    //oclApr.close()
    oclAt.close()
    oclC.close()

    mxC

  }

  private def jvmOuterProdSum(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {

    // This may be already laid out for outer product computation, which may be faster than reorienting
    // both matrices? need to check.
    val (m, n) = (a.nrow, b.ncol)

    // Prefer col-wise result iff a is dense and b is sparse. In all other cases default to row-wise.
    val preferColWiseR = a.getFlavor.isDense && !b.getFlavor.isDense

    val mxR = r.getOrElse {
      (a.getFlavor.isDense, preferColWiseR) match {
        case (false, false) ⇒ b.like(m, n)
        case (false, true) ⇒ b.like(n, m).t
        case (true, false) ⇒ a.like(m, n)
        case (true, true) ⇒ a.like(n, m).t
      }
    }

    // Loop outer products
    if (preferColWiseR) {
      // this means B is sparse and A is not, so we need to iterate over b values and update R columns with +=
      // one at a time.
      for ((acol, brow) ← a.t.zip(b); bel ← brow.nonZeroes) mxR(::, bel.index()) += bel * acol
    } else {
      for ((acol, brow) ← a.t.zip(b); ael ← acol.nonZeroes()) mxR(ael.index(), ::) += ael * brow
    }

    mxR
  }
}