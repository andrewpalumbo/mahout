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

package org.apache.mahout.visualization

import java.awt.{BorderLayout, Color}
import java.io.File
import javax.swing.JFrame

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import smile.plot._

import scala.collection.JavaConversions._


/**
  * Create a s scatter plot of a DRM by sampling a given percentage
  * and plotting corresponding points of (drmXY(::,0),drmXY(::,2))
  *
  * @param drmXY a m x 2 Drm drm to plot
  * @param samplePercent the percentage the drm to sample
  * @tparam K
  */
class mplot2d[K](drmXY: DrmLike[K], samplePercent: Int = 10, setVisible: Boolean = true)  {
     val drmSize = drmXY.numRows()
     val numSamples: Int = (drmSize * (samplePercent / 100))

     println("SampleSize: " + numSamples +"of drm size: "+ drmSize)

     val mPlotMatrix: Matrix = drmSampleKRows(drmXY, 5000, false)
     val arrays: Array[Array[Double]]  = Array.ofDim[Double](mPlotMatrix.numRows(), 2)
     for (i <- 0 until mPlotMatrix.numRows()) {
          arrays(i)(0) = mPlotMatrix(i, 0)
          arrays(i)(1) = mPlotMatrix(i, 1)
     }

     val canvas: PlotCanvas = ScatterPlot.plot(arrays,Color.BLUE)
     canvas.setTitle("2d Plot: " + samplePercent + " % sample of " + drmSize +" points")
     canvas.setAxisLabels("X", "f(x)")

     val plotPanel :PlotPanel = new PlotPanel(canvas)

     val plotFrame: JFrame = new JFrame("2d Plot")
     plotFrame.setLayout(new BorderLayout())
     plotFrame.add(plotPanel)
     plotFrame.setSize(300,300)
     if (setVisible) {
          plotFrame.setVisible(true)
          plotFrame.show()
     }

}
