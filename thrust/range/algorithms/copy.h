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

/// \file \brief Implements the rang-based interface of the copy algorithm

#pragma once

#include <thrust/copy.h>
#include <thrust/range/utilities.h>

namespace thrust
{

/// \addtogroup algorithms

///\addtogroup copying
///  \ingroup algorithms
/// \{

/// \brief Copies the \t SinglePassRange range to the \t OutputRange \p result.
template<typename SinglePassRange, typename OutputRange>
OutputRange& copy(SinglePassRange const& range, OutputRange& result) {
  copy(thrust::begin(range), thrust::end(range), thrust::begin(result));
  return result;
}

/// \}  // end copying

}  // namespace thrust
