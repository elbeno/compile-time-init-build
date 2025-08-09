#pragma once

#include <stdx/compiler.hpp>
#include <stdx/env.hpp>

using cib_log_env_t = stdx::env<>;

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

#define CIB_LOG_ENV_DECL(...)                                                  \
    [[maybe_unused]] typedef decltype([] {                                     \
        return stdx::extend_env_t<cib_log_env_t, __VA_ARGS__>{};               \
    }()) cib_log_env_t

#define CIB_APPEND_LOG_ENV_DECL(E)                                             \
    [[maybe_unused]] typedef decltype([] {                                     \
        return stdx::append_env_t<cib_log_env_t, E>{};                         \
    }()) cib_log_env_t

#define CIB_LOG_ENV(...)                                                       \
    STDX_PRAGMA(diagnostic push)                                               \
    STDX_PRAGMA(diagnostic ignored "-Wshadow")                                 \
    CIB_LOG_ENV_DECL(__VA_ARGS__);                                             \
    STDX_PRAGMA(diagnostic pop)                                                \
    static_assert(true)

#define CIB_APPEND_LOG_ENV(E)                                                  \
    STDX_PRAGMA(diagnostic push)                                               \
    STDX_PRAGMA(diagnostic ignored "-Wshadow")                                 \
    CIB_APPEND_LOG_ENV_DECL(E);                                                \
    STDX_PRAGMA(diagnostic pop)                                                \
    static_assert(true)

#define CIB_WITH_LOG_ENV(...)                                                  \
    STDX_PRAGMA(diagnostic push)                                               \
    STDX_PRAGMA(diagnostic ignored "-Wshadow")                                 \
    if constexpr (CIB_LOG_ENV_DECL(__VA_ARGS__); true)                         \
    STDX_PRAGMA(diagnostic pop)

// NOLINTEND(cppcoreguidelines-macro-usage)
