add_library(${CIB_PREFIX}_log INTERFACE)
target_compile_features(${CIB_PREFIX}_log INTERFACE cxx_std_20)
target_link_libraries_system(${CIB_PREFIX}_log INTERFACE stdx)

target_sources(
    ${CIB_PREFIX}_log
    INTERFACE FILE_SET
              log
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/log/env.hpp
              include/log/flavor.hpp
              include/log/level.hpp
              include/log/log.hpp
              include/log/module.hpp)

add_library(${CIB_PREFIX}_log_fmt INTERFACE)
target_compile_features(${CIB_PREFIX}_log_fmt INTERFACE cxx_std_20)
target_link_libraries_system(${CIB_PREFIX}_log_fmt INTERFACE ${CIB_PREFIX}_log
                             fmt::fmt-header-only stdx)

target_sources(
    ${CIB_PREFIX}_log_fmt
    INTERFACE FILE_SET
              log
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/log/fmt/logger.hpp)

add_library(${CIB_PREFIX}_log_binary INTERFACE)
target_compile_features(${CIB_PREFIX}_log_binary INTERFACE cxx_std_20)
target_link_libraries_system(
    ${CIB_PREFIX}_log_binary INTERFACE ${CIB_PREFIX}_log ${CIB_PREFIX}_msg
    concurrency stdx)

target_sources(
    ${CIB_PREFIX}_log_binary
    INTERFACE FILE_SET
              log
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/log/module_id.hpp
              include/log/string_id.hpp
              include/log/unit.hpp
              include/log/catalog/arguments.hpp
              include/log/catalog/builder.hpp
              include/log/catalog/catalog.hpp
              include/log/catalog/encoder.hpp
              include/log/catalog/mipi_builder.hpp
              include/log/catalog/mipi_messages.hpp
              include/log/catalog/writer.hpp)

if(PROJECT_IS_TOP_LEVEL)
    clang_tidy_interface(${CIB_PREFIX}_log)
    clang_tidy_interface(${CIB_PREFIX}_log_fmt)
    clang_tidy_interface(${CIB_PREFIX}_log_binary)
    add_subdirectory(test)
endif()
