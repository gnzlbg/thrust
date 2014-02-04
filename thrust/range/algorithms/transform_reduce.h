///
///  Copyright 2014 Gonzalo Brito Gadeschi
///
///  Licensed under the Apache License, Version 2.0 (the "License");
///  you may not use this file except in compliance with the License.
///  You may obtain a copy of the License at
///
///  http://www.apache.org/licenses/LICENSE-2.0
///
///  Unless required by applicable law or agreed to in writing, software
///  distributed under the License is distributed on an "AS IS" BASIS,
///  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
///  See the License for the specific language governing permissions and
///  limitations under the License.

/// \file \brief Implements the rang-based interface of the transform_reduce
/// algorithm

#pragma once

#include <thrust/transform_reduce.h>

namespace thrust
{

/// \addtogroup reductions
/// \ingroup transformed_reductions Transformed Reductions
/// \{

/// Reduces the result of applying the \t UnaryFunction \p unary_op to each
/// element of the \t SinglePassRange \p range using the \t BindaryFunction \p
/// binary_op with start value \p init of type \t T.
template<typename SinglePassRange, typename UnaryFunction,
         typename T, typename BinaryFunction>
T transform_reduce(SinglePassRange const& range, UnaryFunction unary_op,
                   T init, BinaryFunction binary_op) {
  using thrust::system::detail::generic::select_system;
  typedef typename SinglePassRange::iterator Iterator;
  typedef typename thrust::iterator_system<Iterator>::type System;
  System system;
  return thrust::transform_reduce(select_system(system), thrust::begin(range),
                                  thrust::end(range), unary_op, init, binary_op);
}

/// \} // end transformed_reductions

} // end namespace thrust
