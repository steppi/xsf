#pragma once

#include <vector>

#include "xsf/error.h"
#include "xsf/stats.h"
// Force defining the parenthesis operator even when compiling with a compiler
// defaulting to C++ >= 23.
#define MDSPAN_USE_PAREN_OPERATOR 1
#include "xsf/third_party/kokkos/mdspan.hpp"

namespace xsf {
namespace numpy {

    namespace detail {
        // Helper to wrap a 1D std::vector in a contiguous mdspan.
        template <typename T>
        auto make_mdspan(std::vector<T>& vec) {
            return std::mdspan<T, std::dextents<ptrdiff_t, 1>>(vec.data(), vec.size());
        }

        template <typename T>
        auto make_mdspan(const std::vector<T>& vec) {
            return std::mdspan<const T, std::dextents<ptrdiff_t, 1>>(vec.data(), vec.size());
        }
    }

    template <typename KMat, typename PMat, typename OutputMat>
    inline void poisson_binom_pmf(KMat k, PMat p, OutputMat out) {
        using T = typename OutputMat::value_type;
        auto n = p.extent(0);
        auto k_size = k.extent(0);
        auto out_size = out.extent(0);
        if (out_size != k_size) {
            set_error("_poisson_binom_pmf", SF_ERROR_MEMORY, "out.shape[-1] must be k.shape[-1]");
            return;
        }
        std::vector<T> pmf(n + 1);
        auto pmf_view = detail::make_mdspan(pmf);
        xsf::poisson_binom_pmf_all(p, pmf_view);
        for (ptrdiff_t i = 0;  i < k.extent(0); i++) {
            out(i) = xsf::take_from_pmf(pmf_view, static_cast<long long int>(k(i)));
        }
    }

    template <typename KMat, typename PMat, typename OutputMat>
    inline void poisson_binom_cdf(KMat k, PMat p, OutputMat out) {
        using T = typename OuputMat::value_type;
        auto n = p.extent(0);
        auto k_size = k.extent(0);
        auto out_size = out.extent(0);
        if (out_size != k_size) {
            set_error("_poisson_binom_cdf", SF_ERROR_MEMORY, "out.shape[-1] must be k.shape[-1]");
            return;
        }
        std::vector<T> cdf(n + 1);
        auto cdf_view = detail::make_mdspan(cdf);
        xsf::poisson_binom_cdf_all(p, cdf_view);
        for (ptrdiff_t i = 0;  i < k.extent(0); i++) {
            out(i) = xsf::take_from_discrete_cdf(cdf_view, static_cast<long long int>(k(i)));
        }
    }

}
}
