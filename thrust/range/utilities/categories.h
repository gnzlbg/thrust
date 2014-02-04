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

}  // namespace thrust
