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

import org.apache.flink.api.scala._
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.CheckpointedFlinkDrm
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.{OpAx, _}
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class AndyTestSuite extends FunSuite with DistributedFlinkSuite {

  test("A blockified") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7) )
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val x: Vector = (0, 1, 2)

    printf("A: InCore = \n%s\n",inCoreA)
    printf("x: Vector = \n%s\n",x)
    val xBcast = drmBroadcast(x)
    A.mapBlock(){
      case (keys, block) => {
        printf("xBcast: Vector = \n%s\n", xBcast.value)

        (keys -> block)
      }
    }

    A.rowSums
  }



}