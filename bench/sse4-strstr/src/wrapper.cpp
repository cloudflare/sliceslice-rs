#include "wrapper.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

#define HAVE_AVX2_INSTRUCTIONS

namespace sse4_strstr {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#include "sse4-strstr/src/all.h"
#pragma GCC diagnostic pop
}  // namespace sse4_strstr

extern "C" {
size_t avx2_strstr_v2(const char* s, size_t n, const char* needle, size_t k) {
  auto result = sse4_strstr::avx2_strstr_v2(s, n, needle, k);

  // Original implementation erroneously assumes a null-terminated haystack in
  // the single-byte needle case so this check must be repeated.
  if (result <= n - k) {
    return result;
  }

  return std::string::npos;
}
}
