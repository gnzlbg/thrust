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

/// \brief Returns a view of the elements of the \t SinglePassRange \p range
/// using the \t UnaryFunction \p op.
template<typename SinglePassRange, typename UnaryFunction>
thrust::transformed_range<SinglePassRange, UnaryFunction>
transform(SinglePassRange const& range, UnaryFunction op)
{ return thrust::make_transformed_range(range, op); }

#if __cplusplus >= 201103L  // \todo C++03 depends on WAR for concept overl.

/// \brief Assing to the \t OutputIterator \p output the result of transforming
/// the \t SinglePassRange \p range using the \t UnaryFunction \p op.
template<typename SinglePassRange, typename OutputIterator,
         typename UnaryFunction>
typename thrust::detail::enable_if<
  !models::single_pass_range<OutputIterator>::value, OutputIterator>::type
transform(SinglePassRange const& range, OutputIterator output, UnaryFunction op) {
  using thrust::system::detail::generic::select_system;
  typedef typename SinglePassRange::iterator Iterator;
  typedef typename thrust::iterator_system<Iterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;
  System1 system1;
  System2 system2;
  return thrust::transform(select_system(system1,system2), thrust::begin(range),
                           thrust::end(range), output, op);
}

#endif  // C++ version >= C++11

/// \brief Assing to the \t SinglePassRange \p output the result of transforming
/// the \t SinglePassRange \p range using the \t UnaryFunction \p op.
template<typename SinglePassRange, typename OutputRange, typename UnaryFunction>
#if __cplusplus >= 201103L
typename thrust::detail::enable_if<
  models::single_pass_range<OutputRange>::value, OutputRange>::type
#else
OutputRange
#endif
transform(SinglePassRange const& range, OutputRange& output, UnaryFunction op) {
  using thrust::system::detail::generic::select_system;
  typedef typename SinglePassRange::iterator Iterator;
  typedef typename thrust::iterator_system<Iterator>::type  System;
  System system;
  thrust::transform(select_system(system), thrust::begin(range),
                    thrust::end(range), output, op);
return output;
}

/// \}  // end modifying transformations

}  // namespace thrust
