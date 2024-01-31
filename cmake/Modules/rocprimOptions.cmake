cmake_dependent_option(ROCPRIM_BUILD_TESTING "Build rocPRIM tests" ON ROCM_BUILD_TESTING OFF)
cmake_dependent_option(ROCPRIM_BUILD_EXAMPLES "Build rocPRIM examples" ON ROCM_BUILD_EXAMPLES OFF)
cmake_dependent_option(ROCPRIM_BUILD_DOCS "Build rocPRIM documentation" ON ROCM_BUILD_DOCS OFF)
cmake_dependent_option(ROCPRIM_BUILD_FILE_REORG_BACKWARD_COMPAT
    "Build rocPRIM with file/folder reorg with backward compatibility enabled" ON
    ROCM_BUILD_FILE_REORG_BACKWARD_COMPAT OFF
)
option(ROCPRIM_BUILD_BENCHMARKS "Build rocPRIM benchmarks" OFF)
option(ROCPRIM_BUILD_CODE_COVERAGE "Build rocPRIM with code coverage enabled" OFF)
option(ROCPRIM_USE_HIP_CPU "Build rocPRIM preferring HIP-CPU runtime instead of HW acceleration" OFF)
