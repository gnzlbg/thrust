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

#include <thrust/transform.h>
#include <thrust/range/utilities.h>
#include <thrust/range/adaptors/transform.h>

namespace thrust {

/// \addtogroup modifying
/// \ingroup transformations
/// \{

/// \brief Returns a view of the elements of the \t SinglePassRange \p range
/// using the \t UnaryFunction \p op.
template<typename SinglePassRange, typename UnaryFunction>
typename thrust::transformed_range<const SinglePassRange, UnaryFunction>::type
transform(SinglePassRange const& range, UnaryFunction op)
{ return thrust::make_transformed_range(range, op); }

/// \brief Assing to the \t SinglePassRange \p output the result of transforming
/// the \t SinglePassRange \p range using the \t UnaryFunction \p op.
template<typename SinglePassRange, typename OutputRange, typename UnaryFunction>
OutputRange& transform(SinglePassRange const& range,
                       OutputRange& output, UnaryFunction op) {
  thrust::transform(thrust::begin(range), thrust::end(range),
                    thrust::begin(output), op);
  return output;
}

/// \}  // end modifying transformations

}  // namespace thrust
