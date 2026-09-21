#pragma once

#include <cassert>
#include <mudock/alpaka_implementation/queue_alpaka.hpp>
#include <mutex>
#include <type_traits>
#include <utility>

namespace mudock {
  template<class F, class... Args>
  inline void queue_alpaka::invoke_kernel(const index3D gridDim, Args&&... args) {
    assert(gridDim.size_x() > 0);
    assert(gridDim.size_y() == 1 && "queue_alpaka currently supports only 1D launches");
    assert(gridDim.size_z() == 1 && "queue_alpaka currently supports only 1D launches");

    const auto grid  = alpaka::Vec<dim, idx>{static_cast<idx>(gridDim.size_x())};
    const auto block = alpaka::Vec<dim, idx>{static_cast<idx>(block_threads())};
    const auto elems = alpaka::Vec<dim, idx>{idx{1}};

    const auto work_div = alpaka::WorkDivMembers<dim, idx>{grid, block, elems};
    auto task           = alpaka::createTaskKernel<acc>(work_div,
                                              F{},
                                              static_cast<std::decay_t<Args>>(std::forward<Args>(args))...);
                                              
    if constexpr (is_kernel_lock_enabled()) {
      auto* lock = alpaka_backend::get_kernel_lock(id, native_device());
      std::unique_lock<std::mutex> guard(lock->mutex);
      if (lock->has_previous_event) {
        alpaka::wait(native_queue(), *(lock->event));
      }
      alpaka::enqueue(native_queue(), task);
      alpaka::enqueue(native_queue(), *(lock->event));
      lock->has_previous_event = true;
    } else {
      alpaka::enqueue(native_queue(), task);
    }
  }

  template<class F, class... Args>
  inline void queue_alpaka::invoke_kernel(const int gridDim, Args&&... args) {
    assert(gridDim >= 0);
    invoke_kernel<F>(index3D{gridDim, 1, 1}, std::forward<Args>(args)...);
  }

  inline constexpr int queue_alpaka::block_threads() { return MUDOCK_ALPAKA_BLOCK_SIZE; }
} // namespace mudock
