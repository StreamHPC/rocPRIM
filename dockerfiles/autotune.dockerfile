# syntax=docker/dockerfile:labs
# The 'ADD <repo>'-instruction is considered experimental

FROM rocm/rocm-terminal:latest

ARG BUILD_DIR="/workspace/rocPRIM/build"
ARG SOURCE_DIR="/workspace/rocPRIM"
ARG REPO_URL="https://github.com/ROCm/rocPRIM.git"
ARG BENCHMARK_TARGETS="benchmark_config_tuning"
ARG GPU_TARGET=""

# Required for apt-add-repository
RUN sudo apt update
RUN sudo apt install software-properties-common ca-certificates gnupg2 wget -y

# Add CMake repository
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -c -s) main"

# Install CMake and ninja build
RUN sudo apt update && sudo apt install ninja-build cmake -y

# Clone repository
ADD ${REPO_URL} ${SOURCE_DIR}
RUN sudo chown --recursive $(id -u -n) ${SOURCE_DIR}
RUN mkdir -p ${BUILD_DIR}

# Configure
RUN cmake \
    -D AMDGPU_TARGETS=${GPU_TARGET} \
    -D BENCHMARK_CONFIG_TUNING:BOOL=ON \
    -D BUILD_BENCHMARK:BOOL=ON \
    -D BUILD_EXAMPLE:BOOL=OFF \
    -D BUILD_TEST:BOOL=ON \
    -D CMAKE_CXX_COMPILER:FILEPATH="/opt/rocm/llvm/bin/clang++" \
    -D CMAKE_BUILD_TYPE:STRING="Release" \
    -G Ninja \
    -S "${SOURCE_DIR}" \
    -B "${BUILD_DIR}"

# Build
RUN cmake --build ${BUILD_DIR} --target $BENCHMARK_TARGETS

# Prepare entrypoint env
ENV AUTOTUNE_RESULT_DIR="/workspace/autotune_results"
ENV AUTOTUNE_FILENAME_REGEX="^benchmark"
ENV AUTOTUNE_ALGORITHM_REGEX=""
ENV AUTOTUNE_SIZE=""
ENV AUTOTUNE_TRIALS=""

ENV SOURCE_DIR=${SOURCE_DIR}
ENV BUILD_DIR=${BUILD_DIR}
ENV GPU_TARGET=${GPU_TARGET}

# Run tuning
ENTRYPOINT rocminfo; mkdir -p $AUTOTUNE_RESULT_DIR; python3 \
    "$SOURCE_DIR/.gitlab/run_benchmarks.py" \
    --benchmark_dir="$BUILD_DIR/benchmark" \
    --benchmark_gpu_architecture="$GPU_TARGET" \
    --benchmark_output_dir="$AUTOTUNE_RESULT_DIR" \
    --benchmark_filename_regex="$AUTOTUNE_FILENAME_REGEX" \
    --benchmark_filter_regex="$AUTOTUNE_ALGORITHM_REGEX" \
    --size="$AUTOTUNE_SIZE" \
    --trials="$AUTOTUNE_TRIALS" \
    --seed=82589933
