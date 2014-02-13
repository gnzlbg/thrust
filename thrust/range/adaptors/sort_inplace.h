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

/// \file \brief Implements sorted adaptor

#pragma once

#include <thrust/range/utilities.h>
#include <thrust/range/iterator_range.h>
#include <thrust/range/adaptors/holder.h>

namespace thrust {

namespace detail {

/// \brief \t StricktWeakOrdering holder for the pipe version of the sort
/// algorithm.
template<typename StricktWeakOrdering>
struct SortHolder : Holder<StricktWeakOrdering> {
  __host__ __device__
  SortHolder(StricktWeakOrdering f) : Holder<StricktWeakOrdering>(f) {}
};

}  // namespace detail

/// \brief Pipe version of the sort algorithm.
template<typename StricktWeakOrdering>
__host__ __device__
detail::SortHolder<StricktWeakOrdering>
sort_inplace(StricktWeakOrdering f)
{ return detail::SortHolder<StricktWeakOrdering>(f); }

/// \brief Pipe overload to apply the transform algorithm
template<typename SinglePassRange, typename StricktWeakOrdering>
__host__ __device__
SinglePassRange&
operator|(SinglePassRange& r,
          detail::SortHolder<StricktWeakOrdering> const& holder) {
  sort(r, holder.value);
  return r;
}

}  // namespace thrust
