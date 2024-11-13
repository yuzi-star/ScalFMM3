// --------------------------------
// See LICENCE file at project root
// File : interpolation/interpolation.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_INTERPOLATION_HPP
#define SCALFMM_INTERPOLATION_INTERPOLATION_HPP

#include "scalfmm/interpolation/uniform/uniform_interpolator.hpp"
#include "scalfmm/interpolation/chebyshev/chebyshev_interpolator.hpp"
#include "scalfmm/options/options.hpp"

namespace scalfmm::interpolation
{
    template<typename ValueType, std::size_t Dimension, typename MatrixKernel, typename Settings>
    struct get_interpolator
    {
        static_assert(options::support(options::_s(Settings{}),
                                       options::_s(options::uniform_dense, options::uniform_low_rank, options::uniform_fft,
                                                   options::chebyshev_dense, options::chebyshev_low_rank)),
                      "unsupported interpolator options!");
    };

    template<typename ValueType, std::size_t Dimension, typename MatrixKernel>
    struct get_interpolator<ValueType, Dimension, MatrixKernel, options::uniform_<options::dense_>>
    {
            using type = uniform_interpolator<ValueType, Dimension, MatrixKernel, options::dense_>;
    };

    template<typename ValueType, std::size_t Dimension, typename MatrixKernel>
    struct get_interpolator<ValueType, Dimension, MatrixKernel, options::uniform_<options::low_rank_>>
    {
            using type = uniform_interpolator<ValueType, Dimension, MatrixKernel, options::low_rank_>;
    };

    template<typename ValueType, std::size_t Dimension, typename MatrixKernel>
    struct get_interpolator<ValueType, Dimension, MatrixKernel, options::uniform_<options::fft_>>
    {
            using type = uniform_interpolator<ValueType, Dimension, MatrixKernel, options::fft_>;
    };

    template<typename ValueType, std::size_t Dimension, typename MatrixKernel>
    struct get_interpolator<ValueType, Dimension, MatrixKernel, options::chebyshev_<options::dense_>>
    {
            using type = chebyshev_interpolator<ValueType, Dimension, MatrixKernel, options::dense_>;
    };

    template<typename ValueType, std::size_t Dimension, typename MatrixKernel>
    struct get_interpolator<ValueType, Dimension, MatrixKernel, options::chebyshev_<options::low_rank_>>
    {
            using type = chebyshev_interpolator<ValueType, Dimension, MatrixKernel, options::low_rank_>;
    };

    template<typename ValueType, std::size_t Dimension, typename MatrixKernel, typename Settings>
    using interpolator = typename get_interpolator<ValueType, Dimension, MatrixKernel, Settings>::type;

}

#endif // SCALFMM_INTERPOLATION_INTERPOLATION_HPP
