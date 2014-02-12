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

/// \file \brief Implements the rang-based interface of the transform algorithm

#pragma once

#include <thrust/detail/type_traits.h>
#include <thrust/range/utilities.h>
#include <thrust/transform.h>
#include <thrust/range/adaptors/transformed.h>

namespace thrust {

/// \addtogroup modifying
/// \ingroup transformations
/// \{

/// \brief Sorts the element in the \t SinglePassRange \p range such that the \t
/// StrictWeakOrdering \p cmp is satisfied (i.e. such that \p cmp evaluates to
/// true for every contiguous pair of elements).
template<typename SinglePassRange,
         typename StrictWeakOrdering
         = thrust::less<typename SinglePassRange::value_type> >
SinglePassRange& sort(SinglePassRange& range,
                      StrictWeakOrdering cmp = StrictWeakOrdering() ) {
  thrust::sort(thrust::begin(range), thrust::end(range), cmp);
  return range;
}

/// \}  // end modifying transformations

}  // namespace thrust
