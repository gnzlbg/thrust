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

/// \file \brief Gets the begin iterator from an iterator range

#pragma once

#if __cplusplus >= 201103L
#include <iterator>
#endif

#include<thrust/range/iterator_range.h>

namespace thrust {

#if __cplusplus >= 201103L

using std::begin;

#else

template<typename SinglePassRange>
typename SinglePassRange::iterator begin(SinglePassRange & range)
{ return range.begin(); }

template<typename SinglePassRange>
typename SinglePassRange::const_iterator begin(SinglePassRange const& range)
{ return range.begin(); }

template<typename T, std::size_t size>
inline T* begin(T (&array)[size])
{ return array; }

template<typename T, std::size_t size>
inline const T* begin(const T (&array)[size])
{ return array; }

#endif

}  // namespace thrust
