package org.apache.mahout.javacpp.linalg;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Platform(includepath={"/usr/include/","/usr/include/CL/","/usr/include/viennacl/"},
 include={"vcl_blas3.h","vcl_blas3.cpp","viennacl/matrix.hpp","viennacl/linalg/prod.hpp","forwards.h",
         "viennacl/detail/matrix_def.hpp"
          })
//@Namespace("vcl_blas3")
public class vcl_blas3 extends Pointer{
    public vcl_blas3(){
        Loader.load();
    }
    static {Loader.load();}

    @Name("dense_dense_mmul")
    public static native void dense_dense_mmul(@Cast("double *") DoublePointer mxA,
                                               @ByVal long mxANrow, @ByVal long mxANcol,
                                               @Cast("double *") DoublePointer mxB,
                                               @ByVal long mxBNrow, @ByVal long mxBNcol,
                                               @Cast("double *") DoublePointer mxRes);
}
