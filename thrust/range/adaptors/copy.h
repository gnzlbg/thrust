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

/// \file \brief Implements copied adaptor

#pragma once

#include <thrust/range/utilities.h>
#include <thrust/range/iterator_range.h>
#include <thrust/range/adaptors/holder.h>

namespace thrust {

namespace detail {

/// \brief \t OutputRange holder for the pipe version of the copy algorithm.
template<typename OutputRange> struct CopiedHolder : Holder<OutputRange> {
  __host__ __device__ CopiedHolder(OutputRange f) : Holder<OutputRange>(f) {}
};

}  // namespace detail

/// \brief Pipe-version of the copy algorithm
template<typename OutputRange>
__host__ __device__ detail::CopiedHolder<OutputRange&> copy(OutputRange& r)
{ return detail::CopiedHolder<OutputRange&>(r); }


/// \brief Pipe operator overload for the copy algorithm
template<typename SinglePassRange, typename OutputRange>
__host__ __device__
OutputRange& operator|(SinglePassRange const& r,
                       detail::CopiedHolder<OutputRange&> const& holder) {
  thrust::copy(r, const_cast<OutputRange&>(holder.value));
  return const_cast<OutputRange&>(holder.value);
}

}  // namespace thrust
