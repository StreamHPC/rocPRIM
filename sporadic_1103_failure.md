- disabling caching fixes it.
- seems like shader can sometimes read from incomplete writes
- s_waitcnt(lgkmcnt==0) doesn't help even though the ISA might say it is required

../../gdb-14.1/gdb/gdb -ex=r --args test/rocprim/test_device_scan --gtest_break_on_failure --gtest_repeat=10000
