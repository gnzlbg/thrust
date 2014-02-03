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

/// \file \brief unqualified type trait

#pragma once

#if __cplusplus == 201103L

#include <type_traits>

namespace thrust {

/// Removes reference and const-volatile qualifiers (as auto i = expr; does)
template <class T>
using unqualified_t = typename std::remove_cv
    <typename std::remove_reference<T>::type>::type;

}  // namespace thrust

#elif __cplusplus > 201103L

namespace thrust {

/// Removes reference and const-volatile qualifiers (as auto i = expr; does)
template <class T>
using unqualified_t = std::remove_cv_t<std::remove_reference_t<T>>;

}  // namespace thrust

#else // C++ < 11

///...

#endif
