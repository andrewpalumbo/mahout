package org.apache.mahout.classifier.naivebayes

/**
 * Created by andy on 8/8/14.
 */

import org.apache.mahout.h2obindings.test.DistributedH2OSuite
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

import org.apache.mahout.test.MahoutSuite
import org.scalatest.FunSuite

class NBSparkTestSuite extends FunSuite with MahoutSuite with DistributedH2OSuite with NBTestBase{
}