// --------------------------------
// See LICENCE file at project root
// File : simd/utils.hpp
// --------------------------------
#ifndef SCALFMM_SIMD_UTILS_HPP
#define SCALFMM_SIMD_UTILS_HPP

#include <xsimd/xsimd.hpp>
#include <cmath>
#include <type_traits>
#include <cstddef>

namespace xsimd
{
    template<class T, class A>
    class batch;
}   // namespace xsimd

namespace scalfmm::simd
{
    template<typename T, class A>
    inline void compare(xsimd::batch<T, A>& x)
    {
        // TODO : assert on 10.*std::numeric_limits<FReal>::epsilon()
        using batch_type = xsimd::batch<T, A>;
        auto booleans = xsimd::abs(x) > batch_type(1.);
        if(xsimd::any(booleans))
        {
            x = xsimd::select(x > batch_type(1.0), batch_type(1.), x);
            x = xsimd::select(x < batch_type(-1.0), batch_type(-1.), x);
        }
    }

    template<typename T>
    inline auto compare(T& x) -> std::enable_if_t<std::is_scalar<T>::value>
    {
        // TODO : assert on 10.*std::numeric_limits<FReal>::epsilon()
        if(std::abs(x) > 1.)
        {
            x = (x > T(1.) ? T(1.) : x);
            x = (x < T(-1.) ? T(-1.) : x);
        }
    }

}   // namespace scalfmm::simd

#endif   // SCALFMM_SIMD_UTILS_HPP
