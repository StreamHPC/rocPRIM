# Stop on first error
set -e

# We require the following environment variables
LIMIT=32M
declare -a SIZES=( 32K 64K 128K 256K 512K 1024K 2048K 4096K 8192K 16384K 32768K )

# ensure results folder exist
mkdir -p ./_results

for SIZE in "${SIZES[@]}";
do
    # Convert to actual numbers
    SIZE_VAL=$(numfmt --from=iec --to=none $SIZE)
    LIMIT_VAL=$(numfmt --from=iec --to=none $LIMIT)
    RESULT_PATH=./_results/device_radix_sort-$LIMIT-$SIZE.json

    echo LIMIT = $LIMIT \( $LIMIT_VAL \)
    echo SIZE  = $SIZE \( $SIZE_VAL \)

    # CMake configure
    cmake -GNinja -DCMAKE_CXX_FLAGS="-DMERGE_SORT_LIMIT=$LIMIT_VAL" -BUILD_BENCHMARK=ON -BUILD_TEST=OF -S .. -B . 

    # Build
    cmake --build . -t benchmark_device_radix_sort

    ./benchmark/benchmark_device_radix_sort --size $SIZE_VAL --name_format json --benchmark_out=$RESULT_PATH --benchmark_out_format=json
done
