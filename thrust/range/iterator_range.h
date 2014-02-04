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

/// \file \brief Implements iterator_range

#pragma once

#include <algorithm>
#include <cassert>
#include <iterator>
#include <thrust/range/utilities/categories.h>
#include <thrust/range/utilities.h>

namespace thrust {

/// \brief Range within a sequence [begin, end)
///
/// Implemented as a pair of iterators
template<typename Iterator> struct iterator_range {
  /// Traits:
  typedef Iterator iterator;
  typedef typename std::iterator_traits<iterator>::value_type value_type;
  typedef typename std::iterator_traits<iterator>::reference reference;
  typedef typename std::iterator_traits<iterator>::difference_type difference_type;
  typedef iterator const_iterator;  // required to satisfy container interface

  #if __cplusplus >= 201103L
  __host__ __device__ iterator_range()                            = default;
  __host__ __device__ iterator_range(iterator_range const&)       = default;
  __host__ __device__ ~iterator_range()                           = default;
  __host__ __device__ iterator_range(iterator_range&&)            = default;
  __host__ __device__ iterator_range& operator=(iterator_range&&) = default;

  // static_asserts:
  /// - for Iterator: move construction should be noexcept(true)
  //  - for iterator_range: move constructor should be noexcept(true)
  # else
  __host__ __device__
  iterator_range() : b_(Iterator()), e_(Iterator()) {}

  __host__ __device__
  iterator_range(iterator_range const& other) : b_(other.b_), e_(other.e_) {}

  __host__ __device__
  ~iterator_range() {}

  __host__ __device__
  iterator_range& operator=(iterator_range const& other) {
    b_ = other.b_;
    e_ = other.e_;
    return *this;
  }
  #endif

  __host__ __device__
  iterator_range(iterator const& b, iterator const& e) : b_(b), e_(e) {}

  /// Interface
  __host__ __device__
  inline iterator    begin() const { return b_;                            }
  __host__ __device__
  inline iterator    end  () const { return e_;                            }
  __host__ __device__
  inline std::size_t size () const { return std::distance(begin(), end()); }
  __host__ __device__
  inline bool        empty() const { return begin() == end();              }

  __host__ __device__
  #if __cplusplus >= 201103L
  explicit
  #endif
  operator bool() const { return !empty(); }

  __host__ __device__
  bool     operator!    () const { return  empty(); }

  __host__ __device__
  reference front() const {
    assert(!empty() && "calling front() on an empty range!");
    return *begin();
  }

  __host__ __device__
  reference back() const {
    assert(!empty() && "calling back() on an empty range!");
    return *(--end());
  }
  __host__ __device__
  void pop_front(difference_type n = 1) {
    assert(!empty() && "calling pop_front() on an empty range!");
    assert(n >= 0);
    std::advance(b_, n);
  }

  __host__ __device__
  void pop_back(difference_type n = -1) {
    assert(!empty() && "calling pop_back() on an empty range!");
    assert(n <= 0);
    std::advance(e_, n);
  }
  __host__ __device__
  reference operator[](difference_type i) const {
    assert(i >= 0 && i < size() && "index out of bounds!");
    return begin()[i];
  }
  __host__ __device__
  reference operator()(difference_type i) const
  { return operator[](i); }

 private:
  Iterator b_;
  Iterator e_;
};


/// Comparison operators
template<typename Iterator>
__host__ __device__
inline bool operator==(iterator_range<Iterator> const& l,
                       iterator_range<Iterator> const& r)
{ return (l.begin() == r.begin()) && (l.end() == r.end()); }

template<typename Iterator>
__host__ __device__
inline bool operator!=(iterator_range<Iterator> const& l,
                       iterator_range<Iterator> const& r)
{ return !(l == r); }

template<typename Iterator>
__host__ __device__
inline bool operator<(iterator_range<Iterator> const& l,
                      iterator_range<Iterator> const& r) {
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  return std::lexicographical_compare
    (l.begin(), l.end(),r.begin(), r.end(),
     std::less<value_type>());
}

template<typename Iterator>
__host__ __device__
inline bool operator>(iterator_range<Iterator> const& l,
                      iterator_range<Iterator> const& r) {
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  return std::lexicographical_compare
    (l.begin(), l.end(), r.begin(), r.end(),
     std::greater<value_type>());
}

template<typename Iterator>
__host__ __device__
inline bool operator<=(iterator_range<Iterator> const& l,
                       iterator_range<Iterator> const& r)
{ return !(l > r); }

template<typename Iterator>
__host__ __device__
inline bool operator>=(iterator_range<Iterator> const& l,
                       iterator_range<Iterator> const& r)
{ return !(l < r); }


/// \brief Returns the \t Iterator range [\p begin, \p end)
template<typename Iterator>
__host__ __device__
iterator_range<Iterator> make_iterator_range(Iterator const& begin,
                                             Iterator const& end)
{ return iterator_range<Iterator>(begin, end); }

}  // namespace thrust
