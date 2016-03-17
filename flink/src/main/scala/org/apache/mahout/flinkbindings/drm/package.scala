package org.apache.mahout.flinkbindings

import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm.{RLikeDrmOps, DrmLike}
import org.apache.mahout.math.drm._
import org.apache.mahout.math._
import RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import RLikeOps._

/**
  * Created by andy on 3/15/16.
  */
package object drm  {

  private[mahout] implicit def bcast2val[K](bcast:FlinkByteBCast[K]):K = bcast.value

  /** Distributed Squared distance matrix computation. */
//def dsqDist(drmX: DrmLike[Int]): DrmLike[Int] = {
//
//    // This is a specific case of pairwise distances of X and Y.
//
//    import RLikeDrmOps._
//
//    // Context needed
//    implicit val ctx = drmX.context
//
//    // Pin to cache if hasn't been pinned yet
//    val drmXcp = drmX.checkpoint()
//
//    // Compute column sum of squares
//    val s = drmXcp ^ 2 rowSums
//
//    val sBcast = drmBroadcast(s)
//
//    (drmXcp %*% drmXcp.t)
//
//      // Apply second part of the formula as per in-core algorithm
//      .mapBlock() { case (keys, block) ⇒
//
//      // Slurp broadcast to memory
//      val s = sBcast: Vector
//
//      // Update in-place
//      block := { (r, c, x) ⇒ s(keys(r)) + s(c) - 2 * x}
//
//      keys → block
//    }
//  }
//
//
//  /**
//    * Compute fold-in distances (distributed version). Here, we use pretty much the same math as with
//    * squared distances.
//    *
//    * D_sq = s*1' + 1*t' - 2*X*Y'
//    *
//    * where s is row sums of hadamard product(X, X), and, similarly,
//    * s is row sums of Hadamard product(Y, Y).
//    *
//    * @param drmX m x d row-wise dataset. Pinned to cache if not yet pinned.
//    * @param drmY n x d row-wise dataset. Pinned to cache if not yet pinned.
//    * @return m x d pairwise squared distance matrix (between rows of X and Y)
//    */
// def dsqDist(drmX: DrmLike[Int], drmY: DrmLike[Int]): DrmLike[Int] = {
//
//    import RLikeDrmOps._
//
//    implicit val ctx = drmX.context
//
//    val drmXcp = drmX.checkpoint()
//    val drmYcp = drmY.checkpoint()
//
//    val sBcast = drmBroadcast(drmXcp ^ 2 rowSums)
//    val tBcast = drmBroadcast(drmYcp ^ 2 rowSums)
//
//    (drmX %*% drmY.t)
//
//      // Apply the rest of the formula
//      .mapBlock() { case (keys, block) =>
//
//      // Cache broadcast representations in local task variable
//      val s = sBcast: Vector
//      val t = tBcast: Vector
//
//      block := { (r, c, x) => s(keys(r)) + t(c) - 2 * x}
//      keys → block
//    }
//  }

}
