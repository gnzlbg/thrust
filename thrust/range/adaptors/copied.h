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

template<typename T> struct CopiedHolder : Holder<T> {
  __host__ __device__ CopiedHolder(T f) : Holder<T>(f) {}
};

}  // namespace detail

template<typename OutputRange>
__host__ __device__
detail::CopiedHolder<OutputRange&> copy(OutputRange& r)
{ return detail::CopiedHolder<OutputRange&>(r); }

template<typename SinglePassRange, typename OutputRange>
__host__ __device__
OutputRange&
operator|(SinglePassRange const& r, detail::CopiedHolder<OutputRange&> const& holder) {
  thrust::copy(r, const_cast<OutputRange&>(holder.value));
  return const_cast<OutputRange&>(holder.value);
}

}  // namespace thrust
