#ifndef BIND_CPU_HPP
#define BIND_CPU_HPP

#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#endif

inline void bind_cpu(int cpu_id = 0)
{
#if defined(_WIN32)
    auto handle = SetThreadAffinityMask(GetCurrentThread(), 1 << cpu_id);
    if (handle == 0)
    {
        std::cout << "Windows bind cpu failed" << std::endl;
    }
    else
    {
        std::cout << "Windows bind to cpu " << cpu_id << std::endl;
    }
#elif defined(__linux__)
    pthread_t tid = pthread_self();

    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);
    CPU_SET(cpu_id, &cpu_mask);
    int ret = pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpu_mask);
    if (ret != 0)
    {
        std::cout << "Linux bind cpu failed" << std::endl;
    }

    cpu_set_t cpu_get;
    pthread_getaffinity_np(tid, sizeof(cpu_set_t), &cpu_get);
    if (CPU_ISSET(cpu_id, &cpu_get))
    {
        std::cout << "Linux bind to cpu " << cpu_id << std::endl;
    }
#endif
}

#endif // BIND_CPU_HPP