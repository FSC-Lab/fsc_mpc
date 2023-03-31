#ifndef QUADROTOR_MPCPP_INTERNAL_HPP_
#define QUADROTOR_MPCPP_INTERNAL_HPP_

#include <memory>
#include <type_traits>

#define CAT_IMPL(A, B) A##B
#define CAT(A, B) CAT_IMPL(A, B)

#define STRINGIFY_IMPL(A) #A
#define STRINGIFY(A) STRINGIFY_IMPL(A)

namespace control {
namespace details {
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

}  // namespace details
}  // namespace control

#endif  // QUADROTOR_MPCPP_INTERNAL_HPP_
