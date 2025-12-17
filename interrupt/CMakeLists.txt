add_library(${CIB_PREFIX}_interrupt INTERFACE)
target_compile_features(${CIB_PREFIX}_interrupt INTERFACE cxx_std_20)
target_link_libraries_system(${CIB_PREFIX}_interrupt INTERFACE concurrency stdx)

target_sources(
    ${CIB_PREFIX}_interrupt
    INTERFACE FILE_SET
              ${CIB_PREFIX}_interrupt
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/interrupt/concepts.hpp
              include/interrupt/config.hpp
              include/interrupt/dynamic_controller.hpp
              include/interrupt/fwd.hpp
              include/interrupt/hal.hpp
              include/interrupt/impl.hpp
              include/interrupt/manager.hpp
              include/interrupt/policies.hpp)

if(PROJECT_IS_TOP_LEVEL)
    clang_tidy_interface(${CIB_PREFIX}_interrupt)
    add_subdirectory(test)
endif()
