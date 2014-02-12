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

#include <thrust/range/utilities.h>
#include <thrust/range/iterator_range.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/range/adaptors/holder.h>

namespace thrust {

template<typename SinglePassRange, typename UnaryFunction>
struct transformed_range
  : iterator_range<transform_iterator<UnaryFunction,
                                      typename SinglePassRange::const_iterator> > {

  typedef iterator_range<
    transform_iterator<UnaryFunction,
                       typename SinglePassRange::const_iterator> > base;

  __host__ __device__
  transformed_range(SinglePassRange const& range, UnaryFunction f)
      : base(make_transform_iterator(begin(range), f),
             make_transform_iterator(end(range), f)) {}
};

template<typename SinglePassRange, typename UnaryFunction>
__host__ __device__
transformed_range<const SinglePassRange, UnaryFunction>
make_transformed_range(SinglePassRange const& r, UnaryFunction f)
{ return transformed_range<const SinglePassRange, UnaryFunction>(r, f); }

namespace detail {

template<typename T> struct TransformHolder : Holder<T> {
  __host__ __device__ TransformHolder(T f) : Holder<T>(f) {}
};

}  // namespace detail

template<typename UnaryFunction>
__host__ __device__
detail::TransformHolder<UnaryFunction> transformed(UnaryFunction f)
{ return detail::TransformHolder<UnaryFunction>(f); }

template<typename SinglePassRange, typename UnaryFunction>
__host__ __device__
transformed_range<const SinglePassRange, UnaryFunction>
operator|(SinglePassRange const& r, detail::TransformHolder<UnaryFunction> const& holder)
{ return make_transformed_range(r, holder.value); }

}  // namespace thrust
