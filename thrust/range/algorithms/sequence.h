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

/// \file \brief Implements the rang-based interface of the sequence algorithm

#pragma once

#include <thrust/sequence.h>

namespace thrust {

/// \brief Initializes the \t SinglePassRange \p range with the sequence \p init
/// + 0 * \p step, \p init + 1 * \p step, ..., \p init + (distance(range) - 1) *
/// \p step.
template<typename SinglePassRange, typename T>
SinglePassRange& sequence(SinglePassRange& range, T init = T(), T step = T(1)) {
  thrust::sequence(thrust::begin(range), thrust::end(range), init, step);
  return range;
}

} // namespace thrust
