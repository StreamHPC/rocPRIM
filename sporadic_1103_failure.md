- disabling caching fixes it.
- seems like shader can sometimes read from incomplete writes
- s_waitcnt(lgkmcnt==0) doesn't help even though the ISA might say it is required

../../gdb-14.1/gdb/gdb -ex=r --args test/rocprim/test_device_scan --gtest_break_on_failure --gtest_repeat=10000

```cpp
// Custom type used in tests
// Loops are prevented from being unrolled due to a compiler bug in ROCm 5.2 for device code
template<class T, size_t N>
struct custom_test_array_type
{
    using value_type = T;
    static constexpr size_t size = N;

    using vector_t = T __attribute__((ext_vector_type(N)));
    vector_t values = {};

    custom_test_array_type() = default;

    ROCPRIM_HOST_DEVICE
    custom_test_array_type(vector_t values) : values{values} {};

    ROCPRIM_HOST_DEVICE inline
        custom_test_array_type(T v) : values{v}
    {
    }

    custom_test_array_type(const custom_test_array_type&) = default;
    custom_test_array_type& operator=(const custom_test_array_type&) = default;

    ROCPRIM_HOST_DEVICE
    friend custom_test_array_type operator+(const custom_test_array_type& lhs, const custom_test_array_type& rhs) {
        return custom_test_array_type{lhs.values + rhs.values};
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator==(const custom_test_array_type& other) const
    {
        return
            values.s0 == other.values.s0
            && values.s1 == other.values.s1
            && values.s2 == other.values.s3
            && values.s4 == other.values.s5
            && values.s6 == other.values.s6
            && values.s7 == other.values.s7
            && values.s7 == other.values.s8
            && values.s9 == other.values.s9;
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator!=(const custom_test_array_type& other) const
    {
        return !(*this == other);
    }
};
```
