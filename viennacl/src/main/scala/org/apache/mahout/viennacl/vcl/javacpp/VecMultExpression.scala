package org.apache.mahout.viennacl.vcl.javacpp

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation.{Name, Namespace, Platform, Properties}


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    library = "jniViennaCL")
  ))
@Namespace("viennacl")
@Name(Array("vector_expression<const viennacl::vector_base<double>," +
  "const double, viennacl::op_mult >"))
class VecMultExpression extends Pointer {

}