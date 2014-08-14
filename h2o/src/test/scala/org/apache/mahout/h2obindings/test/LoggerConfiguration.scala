package org.apache.mahout.h2obindings.test

import org.scalatest.{ConfigMap, Suite}
import org.apache.log4j.{Level, Logger, BasicConfigurator}

trait LoggerConfiguration extends org.apache.mahout.test.LoggerConfiguration {
  this: Suite =>

  override protected def beforeAll(configMap: ConfigMap) {
    super.beforeAll(configMap)
    Logger.getLogger("org.apache.mahout.h2obindings").setLevel(Level.DEBUG)
  }
}
