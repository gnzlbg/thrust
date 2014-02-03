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

template<typename T>
inline typename iterator_range<T>::iterator end(T& r)
{ return r.end(); }

template<typename T>
inline typename iterator_range<const T>::iterator end(const T& r)
{ return r.end(); }

template<typename T, std::size_t size>
inline T* end(T (&array)[size])
{ return array + size; }

template<typename T, std::size_t size>
inline const T* end(const T (&array)[size])
{ return array + size; }

#endif

}  // namespace thrust
