// Macros and internal functions and type definitions
// Copyright © 2023 FSC Lab
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FSC_MPC_INTERNAL_HPP_
#define FSC_MPC_INTERNAL_HPP_

#include <memory>
#include <type_traits>

#define CAT_IMPL(A, B) A##B
#define CAT(A, B) CAT_IMPL(A, B)

#define STRINGIFY_IMPL(A) #A
#define STRINGIFY(A) STRINGIFY_IMPL(A)

namespace fsc::details {
template <typename T>
constexpr auto MutData(const T &obj) -> std::add_pointer_t<
    std::remove_const_t<std::remove_pointer_t<decltype(obj.data())>>> {
  using ConstElement = std::remove_pointer_t<decltype(obj.data())>;
  using MutPtr = std::add_pointer_t<std::remove_const_t<ConstElement>>;
  return const_cast<MutPtr>(obj.data());
}

template <typename T>
constexpr std::underlying_type_t<T> ToUnderlying(T value) {
  return static_cast<std::underlying_type_t<T>>(value);
}

template <typename T, int (*D)(T *)>
struct DeleterWrapper {
  inline void operator()(T *obj) const { static_cast<void>(D(obj)); }
};

template <typename T, int (*D)(T *)>
using Handle = std::unique_ptr<T, DeleterWrapper<T, D>>;

}  // namespace fsc::details

#endif  // FSC_MPC_INTERNAL_HPP_
