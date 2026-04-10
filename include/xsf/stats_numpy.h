#pragma once

#include "xsf/error.h"
#include "xsf/stats.h"
// Force defining the parenthesis operator even when compiling with a compiler
// defaulting to C++ >= 23.
#define MDSPAN_USE_PAREN_OPERATOR 1
#include "xsf/third_party/kokkos/mdspan.hpp"

namespace xsf {
namespace numpy {

    template <typename KMat, typename PMat, typename OutputMat>
    inline void poisson_binom_pmf(KMat k, PMat p, OutMat out) {
        using T = typename OuputMat::value_type;
        auto n = p.extent(0);
        auto k_size = k.extent(0);
        auto out_size = out.extent(0);
        if (out_size != k_size) {
            set_error("_poisson_binom_pmf", SF_ERROR_MEMORY, "out.shape[-1] must be k.shape[-1]");
            return;
        }
        std::vector<T> pmf(n + 1);
        std::mdspan<T, std::dextents<ptrdiff_t, 1>> pmf_view(pmf.data(), n + 1);
        xsf::poisson_binom_pmf_all(p, pmf_view)
        for (ptrdiff_t i = 0;  i < k.extent(0); i++) {
            out(i) = xsf::take_from_pmf(pmf_view, static_cast<long long int>(k(i)));
        }
    }

    template <typename KMat, typename PMat, typename OutputMat>
    inline void poisson_binom_cdf(KMat k, PMat p, OutMat out) {
        using T = typename OuputMat::value_type;
        auto n = p.extent(0);
        auto k_size = k.extent(0);
        auto out_size = out.extent(0);
        if (out_size != k_size) {
            set_error("_poisson_binom_cdf", SF_ERROR_MEMORY, "out.shape[-1] must be k.shape[-1]");
            return;
        }
        std::vector<T> cdf(n + 1);
        std::mdspan<T, std::dextents<ptrdiff_t, 1>> cdf_view(cdf.data(), n + 1);
        xsf::poisson_binom_cdf_all(p, cdf_view)
        for (ptrdiff_t i = 0;  i < k.extent(0); i++) {
            out(i) = xsf::take_from_discrete_cdff(cdf_view, static_cast<long long int>(k(i)));
        }
    }

}
}
