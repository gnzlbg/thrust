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

namespace thrust {

/// \addtogroup modifying
/// \ingroup transformations
/// \{

/// \brief Transforms the elements of the \t SinglePassRange \p range in place
/// using the \t UnaryFunction \p op.
template<typename SinglePassRange, typename UnaryFunction>
SinglePassRange transform(SinglePassRange& range, UnaryFunction op) {
  using thrust::system::detail::generic::select_system;
  typedef typename SinglePassRange::iterator Iterator;
  typedef typename thrust::iterator_system<Iterator>::type  System;
  System system;
  thrust::transform(select_system(system), range.begin(), range.end(),
                    range.begin(), op);
  return range;
}

/// \brief Assing to the \t OutputIterator \p output the result of transforming
/// the \t SinglePassRange \p range using the \t UnaryFunction \p op.
template<typename SinglePassRange, typename OutputIterator,
         typename UnaryFunction>
OutputIterator transform(SinglePassRange const& range, OutputIterator output,
                         UnaryFunction op) {
  using thrust::system::detail::generic::select_system;
  typedef typename SinglePassRange::iterator Iterator;
  typedef typename thrust::iterator_system<Iterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;
  System1 system1;
  System2 system2;
  return thrust::transform(select_system(system1,system2), range.begin(),
                           range.end(), output, op);
}

/// \brief Assing to the \t SinglePassRange \p output the result of transforming
/// the \t SinglePassRange \p range using the \t UnaryFunction \p op.
template<typename SinglePassRange, typename UnaryFunction>
SinglePassRange transform(SinglePassRange const& range, SinglePassRange& output,
                          UnaryFunction op) {
  using thrust::system::detail::generic::select_system;
  typedef typename SinglePassRange::iterator Iterator;
  typedef typename thrust::iterator_system<Iterator>::type  System;
  System system;
  thrust::transform(select_system(system), range.begin(), range.end(), output,
                    op);
  return output;
}


/*! \} // end modifying transformations
 */

} // end namespace thrust
