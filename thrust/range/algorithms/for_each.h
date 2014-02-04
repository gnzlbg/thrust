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

/// \file \brief Implements the rang-based interface of the for_each algorithm

#pragma once

#include <thrust/for_each.h>

namespace thrust {

/// \addtogroup non-modifying
/// \ingroup transformations
/// \{

/// \brief Applies \t UnaryFunction \p op to each element of the \t
/// SinglePassRange \p range.
template<typename SinglePassRange, typename UnaryFunction>
SinglePassRange for_each(SinglePassRange const& range, UnaryFunction op) {
  using thrust::system::detail::generic::select_system;
  typedef typename SinglePassRange::iterator Iterator;
  typedef typename thrust::iterator_system<Iterator>::type System;

  System system;
  thrust::for_each(select_system(system), thrust::begin(range),
                   thrust::end(range), op);
  return range;
}

/// \}  // end non-modifying transformations

} // end namespace thrust
