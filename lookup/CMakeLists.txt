add_library(${CIB_PREFIX}_lookup INTERFACE)
target_compile_features(${CIB_PREFIX}_lookup INTERFACE cxx_std_20)
target_link_libraries_system(${CIB_PREFIX}_lookup INTERFACE stdx)

target_sources(
    ${CIB_PREFIX}_lookup
    INTERFACE FILE_SET
              ${CIB_PREFIX}_lookup
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/lookup/detail/select.hpp
              include/lookup/entry.hpp
              include/lookup/input.hpp
              include/lookup/linear_search_lookup.hpp
              include/lookup/lookup.hpp
              include/lookup/pseudo_pext_lookup.hpp
              include/lookup/strategies.hpp
              include/lookup/strategy_failed.hpp)

if(PROJECT_IS_TOP_LEVEL)
    clang_tidy_interface(${CIB_PREFIX}_lookup)
    add_subdirectory(test)
    add_subdirectory(benchmark)
endif()
