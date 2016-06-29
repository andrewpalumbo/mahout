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

#include <stdlib.h>
#include "vcl_blas3.h"

#include "viennacl/matrix.hpp"
//#include "viennacl/detail/matrix_def.hpp"
//#include "viennacl/compressed_matrix.hpp"
//#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "forwards.h"
//#include "mem_handle.hpp"


namespace vcl_blas3 {
    // bridge to JNI functions

    // dense matrices BLAS-3
    // dense %*% dense
     void dense_dense_mmul(double* lhs, long lhs_rows, long lhs_cols,
                           double* rhs, long rhs_rows, long rhs_cols,
                           double* result) {
    /** VCL 1.5.2
    * explicit matrix_base(SCALARTYPE * ptr_to_mem, viennacl::memory_types mem_type,
    *           size_type mat_size1, size_type mat_start1, difference_type mat_stride1, size_type mat_internal_size1,
    *           size_type mat_size2, size_type mat_start2, difference_type mat_stride2, size_type mat_internal_size2)
    */
     viennacl::matrix_base<double> mx_a(lhs, viennacl::MAIN_MEMORY, lhs_rows, 0, 1, lhs_rows,
                                                                      lhs_cols, 0, 1, lhs_cols);
     viennacl::matrix_base<double> mx_b(lhs, viennacl::MAIN_MEMORY, rhs_rows, 0, 1, rhs_rows,
                                                                    rhs_cols, 0, 1, rhs_cols);
     // resulting matrix
     viennacl::matrix_base<double> res(result, viennacl::MAIN_MEMORY, lhs_rows, 0, 1, lhs_rows,
                                                                        rhs_cols, 0, 1, rhs_cols);

     res = viennacl::linalg::prod(mx_a, mx_b);


     /** VCL 1.7.1
      * explicit matrix(NumericT * ptr_to_mem, viennacl::memory_types mem_type,
      *                  size_type rows, size_type cols)
      *
      * viennacl:: matrix_base<double> mx_a(lhs, viennacl::MAIN_MEMORY, lhs_rows, lhs_cols);
      * viennacl:: matrix_base<double> mx_b(rhs, viennacl::MAIN_MEMORY, rhs_rows, rhs_cols);
      *
      * // resulting matrix
      * viennacl:: matrix_base<double> res(result, viennacl::MAIN_MEMORY, lhs_rows, rhs_cols);
      *
      * res = viennacl::linalg::prod(mx_a, mx_b);
      *
      */



    }
}


