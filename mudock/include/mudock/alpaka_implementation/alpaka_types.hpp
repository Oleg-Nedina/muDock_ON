#pragma once

#include <alpaka/alpaka.hpp>
#include <mutex>
#include <memory>
#include <cstddef>

namespace mudock::alpaka_backend {
  using dim = alpaka::DimInt<1u>;
  using idx = std::size_t;

#if defined(MUDOCK_ALPAKA_BACKEND_SERIAL)
  using acc = alpaka::AccCpuSerial<dim, idx>;
#elif defined(MUDOCK_ALPAKA_BACKEND_THREADS)
  using acc = alpaka::AccCpuThreads<dim, idx>;
#elif defined(MUDOCK_ALPAKA_BACKEND_TBB)
  using acc = alpaka::AccCpuTbbBlocks<dim, idx>;
#elif defined(MUDOCK_ALPAKA_BACKEND_OMP2)
  using acc = alpaka::AccCpuOmp2Blocks<dim, idx>;
#elif defined(MUDOCK_ALPAKA_BACKEND_CUDA)
  using acc = alpaka::AccGpuCudaRt<dim, idx>;
#elif defined(MUDOCK_ALPAKA_BACKEND_HIP)
  using acc = alpaka::AccGpuHipRt<dim, idx>;
#elif defined(MUDOCK_ALPAKA_BACKEND_SYCL)
  using acc = alpaka::AccGpuSyclIntel<dim, idx>;
#else
  #error "MUDOCK_ALPAKA_BACKEND_* compile definition is required when MUDOCK_USE_ALPAKA is enabled"
#endif

  using dev_acc   = alpaka::Dev<acc>;
  using queue_acc = alpaka::Queue<acc, alpaka::NonBlocking>;
  using event_acc = alpaka::Event<queue_acc>;

  struct device_kernel_lock {
    std::mutex mutex;
    std::unique_ptr<event_acc> event;
    bool has_previous_event{false};
  };

  device_kernel_lock* get_kernel_lock(int dev_id, const dev_acc& dev);
} // namespace mudock::alpaka_backend

constexpr bool is_kernel_lock_enabled() {
#ifdef MUDOCK_KERNEL_LOCK
  return true;
#else
  return false;
#endif
}
