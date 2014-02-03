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

/// \file \brief Implements zip iterator adaptor

#pragma once

#include <thrust/range/utilities.h>
#include <thrust/range/iterator_range.h>

namespace thrust {

#if __cplusplus >= 201103L

namespace detail {

template<typename... InputIterators>
auto make_zip_iterator_(InputIterators... inputIterators)
RETURNS(zip_iterator<tuple<InputIterators...>>{make_tuple(inputIterators...)});

}  // namespace detail

template<typename... SinglePassRanges>
auto zip(SinglePassRanges&&... ranges)
RETURNS(make_iterator_range(detail::make_zip_iterator_(begin(ranges)...),
                            detail::make_zip_iterator_(end(ranges)...)));

#else  // C++ < C++11

namespace detail {

template<typename InputIterator0>
zip_iterator<tuple<InputIterator0> > make_zip_iterator_(InputIterator0 it0)
{ return make_tuple(it0); }

template<typename InputIterator0, typename InputIterator1>
zip_iterator<tuple<InputIterator0, InputIterator1> >
make_zip_iterator_(InputIterator0 it0, InputIterator1 it1)
{ return make_tuple(it0, it1); }

template<typename InputIterator0, typename InputIterator1,
         typename InputIterator2>
zip_iterator<tuple<InputIterator0, InputIterator1, InputIterator2> >
make_zip_iterator_(InputIterator0 it0, InputIterator1 it1, InputIterator2 it2)
{ return make_tuple(it0, it1, it2); }

template<typename InputIterator0, typename InputIterator1,
         typename InputIterator2, typename InputIterator3>
zip_iterator<tuple<InputIterator0, InputIterator1, InputIterator2,
                   InputIterator3> >
make_zip_iterator_(InputIterator0 it0, InputIterator1 it1, InputIterator2 it2,
                  InputIterator3 it3)
{ return make_tuple(it0, it1, it2, it3); }

}  // namespace detail

template<typename SinglePassRange0>
iterator_range<zip_iterator<tuple<typename SinglePassRange0::iterator> > >
zip(SinglePassRange0 range0) {
  return make_iterator_range(detail::make_zip_iterator_(range0.begin()),
                             detail::make_zip_iterator_(range0.end()));
}

template<typename SinglePassRange0, typename SinglePassRange1>
iterator_range<zip_iterator<tuple<typename SinglePassRange0::iterator,
                                  typename SinglePassRange1::iterator> > >
zip(SinglePassRange0 range0, SinglePassRange1 range1) {
  return make_iterator_range(detail::make_zip_iterator_(range0.begin(),
                                                        range1.begin()),
                             detail::make_zip_iterator_(range0.end(),
                                                        range1.end()));
}

template<typename SinglePassRange0, typename SinglePassRange1,
         typename SinglePassRange2>
iterator_range<zip_iterator<tuple<typename SinglePassRange0::iterator,
                                  typename SinglePassRange1::iterator,
                                  typename SinglePassRange2::iterator> > >
zip(SinglePassRange0 range0, SinglePassRange1 range1,
      SinglePassRange2 range2) {
  return make_iterator_range(detail::make_zip_iterator_(range0.begin(),
                                                        range1.begin(),
                                                        range2.begin()),
                             detail::make_zip_iterator_(range0.end(),
                                                        range1.end(),
                                                        range2.end()));
}

template<typename SinglePassRange0, typename SinglePassRange1,
         typename SinglePassRange2, typename SinglePassRange3>
iterator_range<zip_iterator<tuple<typename SinglePassRange0::iterator,
                                  typename SinglePassRange1::iterator,
                                  typename SinglePassRange2::iterator,
                                  typename SinglePassRange3::iterator> > >
zip(SinglePassRange0 range0, SinglePassRange1 range1,
      SinglePassRange2 range2, SinglePassRange3 range3) {
  return make_iterator_range(detail::make_zip_iterator_(range0.begin(),
                                                        range1.begin(),
                                                        range2.begin(),
                                                        range3.begin()),
                             detail::make_zip_iterator_(range0.end(),
                                                        range1.end(),
                                                        range2.end(),
                                                        range3.end()));
}

#endif
}  // namespace thrust
