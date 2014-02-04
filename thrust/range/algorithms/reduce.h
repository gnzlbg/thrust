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

/// \file \brief Implements the rang-based interface of the reduce algorithm

#pragma once

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/range/utilities.h>

namespace thrust {

/// \addtogroup reductions
/// \{

/// \brief Reduces the \t SinglePassRange \p range with the \t BinaryFunction \p
/// binary_op to a value of type \t T using \p init as the initial reduction
/// value.
template<typename SinglePassRange,
         typename T = typename SinglePassRange::value_type,
         typename BinaryFunction = plus<T> >
T reduce(SinglePassRange const& range, T init = T(),
         BinaryFunction binary_op = BinaryFunction()) {
  using thrust::system::detail::generic::select_system;
  typedef typename SinglePassRange::iterator Iterator;
  typedef typename thrust::iterator_system<Iterator>::type System;
  System system;
  return thrust::reduce(select_system(system), begin(range), end(range),
                        init, binary_op);
}

/// \}  // end reductions

} // end namespace thrust
