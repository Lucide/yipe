#pragma once

#include "common.h"

namespace hash_table {
inline constexpr u8 blocks_per_sm = 2;
inline constexpr u16 threads_per_block = 256;
inline constexpr u16 items_per_thread_per_slice = 8;
} // namespace hash_table
