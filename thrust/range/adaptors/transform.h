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

/// \file \brief Implements zip iterator adaptor

#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/range/utilities.h>
#include <thrust/range/iterator_range.h>
#include <thrust/range/adaptors/holder.h>

namespace thrust {

namespace detail {

/// \brief \t UnaryFunction holder for the pipe version of the transform
/// algorithm.
template<typename UnaryFunction>
struct TransformHolder : Holder<UnaryFunction> {
  __host__ __device__
  TransformHolder(UnaryFunction f) : Holder<UnaryFunction>(f) {}
};

}  // namespace detail

/// \brief Metafunction for getting the type of the transformed range
template<typename SinglePassRange, typename UnaryFunction>
struct transformed_range {
  typedef iterator_range<
    transform_iterator<
      UnaryFunction,
      typename SinglePassRange::const_iterator> > type;
};

/// \brief Transforms the \t SinglePassRange \p r using the \t UnaryFunction \p
/// f.
template<typename SinglePassRange, typename UnaryFunction>
__host__ __device__
typename transformed_range<SinglePassRange, UnaryFunction>::type
make_transformed_range(SinglePassRange const& r, UnaryFunction f) {
  return make_iterator_range(
      make_transform_iterator(thrust::begin(r), f),
      make_transform_iterator(thrust::end(r), f));
}

/// \brief Pipe version of the transform algorithm
template<typename UnaryFunction>
__host__ __device__
detail::TransformHolder<UnaryFunction> transform(UnaryFunction f)
{ return detail::TransformHolder<UnaryFunction>(f); }

/// \brief Pipe overload to apply the transform algorithm
template<typename SinglePassRange, typename UnaryFunction>
__host__ __device__
typename transformed_range<SinglePassRange, UnaryFunction>::type
operator|(SinglePassRange const& r,
          detail::TransformHolder<UnaryFunction> const& holder)
{ return make_transformed_range(r, holder.value); }

}  // namespace thrust
