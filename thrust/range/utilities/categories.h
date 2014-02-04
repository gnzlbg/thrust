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

/// \file \brief Range categories and concept checking utilities

#pragma once

namespace thrust {

namespace range_tags {

struct SinglePassRange {};
struct InputRange         : SinglePassRange         {};
struct OutputRange        : SinglePassRange         {};
struct ForwardRange       : InputRange, OutputRange {};
struct BidirectionalRange : ForwardRange            {};
struct RandomAccessRange  : BidirectionalRange      {};

}  // namespace range_tags

#if __cplusplus >= 201103L  // \todo Provide C++03 workaround for this:

namespace models {

template<class T> struct has_begin {
  template<class U>
  static auto test(U* u) -> decltype(u->begin(), std::true_type());
  template<class U>
  static auto test(...) -> std::false_type;
  static const bool value = decltype(test<T>(0))();
};

template<class T> struct has_end {
  template<class U>
  static auto test(U* u) -> decltype(u->end(), std::true_type());
  template<class U>
  static auto test(...) -> std::false_type;
  static const bool value = decltype(test<T>(0))();
};

template<class T> struct single_pass_range {
  static const bool value = has_begin<T>::value
                            && has_end<T>::value;
};

}  // namespace models

#endif  // C++ version >= C++11

}  // namespace thrust
