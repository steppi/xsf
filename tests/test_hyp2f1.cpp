#include <complex>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

#include <xsf/fp_error_metrics.h>
#include <xsf/hyp2f1.h>

#include "testing_utils.h"

namespace fs = std::filesystem;

fs::path hyp2f1_tables_path{fs::path(XSREF_TABLES_PATH) / "scipy_special_tests" / "hyp2f1"};

TEST_CASE("hyp2f1 complex scipy.special cases", "[hyp2f1][complex][scipy-special]") {
    auto [input, output, tol] = GENERATE(
        xsf_test_cases<
            std::tuple<double, double, double, std::complex<double>>, std::tuple<std::complex<double>, bool>, double>(
            hyp2f1_tables_path / "In_d_d_d_cd-cd.parquet", hyp2f1_tables_path / "Out_d_d_d_cd-cd.parquet",
            hyp2f1_tables_path / "Err_d_d_d_cd-cd_gcc-linux-x86_64.parquet"
        )
    );
    auto [a, b, c, z] = input;
    auto [desired, fallback] = output;
    auto out = xsf::hyp2f1(a, b, c, z);
    auto error = xsf::extended_relative_error(out, desired);
    INFO("a := " << std::setprecision(std::numeric_limits<double>::max_digits10) << a << '\n'
	 << "b := " << b << '\n'
	 << "c := " << c << '\n'
	 << "z := " << z << '\n'
	 << "out := " << out << '\n'
	 << "desired := " << desired << '\n'
	 << "error := " << error << '\n'
	 << "tolerance := " << 2*tol << '\n'
	);
    REQUIRE(error < 2*tol);
}
