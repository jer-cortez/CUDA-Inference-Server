// DeviceBuffer is entirely header-only (cuda_db/memory/device_buffer.hpp) --
// its methods are small enough that out-of-line definitions would just add
// an extra call for no benefit. This translation unit exists so the header
// gets compiled at least once even by a build that never instantiates a
// DeviceBuffer directly, catching syntax errors in CI before any caller
// does.
#include "cuda_db/memory/device_buffer.hpp"
