package org.apache.mahout.javacpp.linalg;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Platform(includepath={"/usr/include/","/usr/include/CL/","/usr/include/viennacl/"},
 include={"vcl_blas3.h","viennacl/matrix.hpp","viennacl/linalg/prod.hpp","forwards.h" })
//@Namespace("mmul")
public class vcl_blas3 {
    public vcl_blas3(){
        Loader.load();
    }
    static {Loader.load();}

    @Name("dense_dense_mmul")
    public static native void dense_dense_mmul(@Cast("double* ") double[] mxA,
                                               long mxANrow, long mxANcol,
                                               @Cast("double* ") double[] mxB,
                                               long mxBNrow, long mxBNcol,
                                               @Cast("double* ") double[] mxRes);
}
//#include "viennacl/matrix.hpp"
//        #include "viennacl/compressed_matrix.hpp"
//        #include "viennacl/coordinate_matrix.hpp"
//        #include "viennacl/linalg/prod.hpp"
//        #include "forwards.h"
//        #include "mem_handle.hpp"