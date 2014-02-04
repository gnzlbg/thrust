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

/// \file \brief Implements a counting_range

#pragma once

namespace thrust {

template<class Integral>
__host__ __device__
iterator_range<thrust::counting_iterator<Integral> >
make_counting_range(Integral const& b, Integral const& e) {
  return thrust::make_iterator_range(
      thrust::counting_iterator<Integral>(b),
      thrust::counting_iterator<Integral>(e));
}

}  // namespace thrust
