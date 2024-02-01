# MIT License
#
# Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
cmake_policy(VERSION 3.16...3.28)
enable_testing()

project(rocprim LANGUAGES CXX)

find_package(ROCmCMakeBuildTools REQUIRED)
include(ROCmDefaults)
include(ROCmOptions)
include(${PROJECT_NAME}Options)
include(${PROJECT_NAME}Dependencies)

include(cmake/VerifyCompiler.cmake)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# Set CXX flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF) # don't build client dependencies as shared

if(ROCPRIM_BUILD_FILE_REORG_BACKWARD_COMPAT)
  rocm_wrap_header_dir(
    "${PROJECT_SOURCE_DIR}/rocprim/include/rocprim"
    WRAPPER_LOCATIONS rocprim/include/rocprim
    OUTPUT_LOCATIONS rocprim/wrapper/include/rocprim
    PATTERNS *.hpp
  )
endif()

if(ROCPRIM_BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

# 'set(VERSION_STRING' is searched for by docs (conf.py)
set(VERSION_STRING "3.0.0")
rocm_setup_version(VERSION ${VERSION_STRING})

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# rocPRIM library
add_subdirectory(rocprim)

if(ROCPRIM_BUILD_TESTING OR ROCPRIM_BUILD_BENCHMARKS)
  rocm_package_setup_component(clients)
endif()

# Tests
if(ROCPRIM_BUILD_TESTING)
  rocm_package_setup_client_component(tests)
  add_subdirectory(test)
endif()

# Benchmarks
if(ROCPRIM_BUILD_BENCHMARKS)
  rocm_package_setup_client_component(benchmarks)
  add_subdirectory(benchmark)
endif()

# Examples
if(ROCPRIM_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

# Docs
if(ROCPRIM_BUILD_DOCS)
  add_subdirectory(docs)
endif()

# Package
set(BUILD_SHARED_LIBS ON) # Build as though shared library for naming
rocm_package_add_dependencies(DEPENDS "hip-rocclr >= 3.5.0")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

rocm_create_package(
  NAME rocprim
  DESCRIPTION "Radeon Open Compute Parallel Primitives Library"
  MAINTAINER "rocPRIM Maintainer <rocprim-maintainer@amd.com>"
  HEADER_ONLY
)


#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(ROCPRIM_BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > # run your tests
  #  > make coverage (generate html documentation)
  #

  #
  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  #
  add_custom_target(coverage
    DEPENDS rocprim
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
    )

  #
  # Generate coverage output.
  #
  add_custom_command(TARGET coverage
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
    )

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
