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

/// \file \brief Gets the end iterator from an iterator range

#if __cplusplus >= 201103L
#include <iterator>
#endif

#pragma once

namespace thrust {

#if __cplusplus >= 201103L

using std::end;

#else

template<typename SinglePassRange>
__host__ __device__
typename SinglePassRange::iterator end(SinglePassRange& range)
{ return range.end(); }

template<typename SinglePassRange>
__host__ __device__
typename SinglePassRange::const_iterator end(SinglePassRange const& range)
{ return range.end(); }

template<typename T, std::size_t size>
__host__ __device__
inline T* end(T (&array)[size])
{ return array + size; }

template<typename T, std::size_t size>
__host__ __device__
inline const T* end(const T (&array)[size])
{ return array + size; }

#endif

}  // namespace thrust
