add_library(${CIB_PREFIX}_msg INTERFACE)
target_compile_features(${CIB_PREFIX}_msg INTERFACE cxx_std_20)
target_link_libraries_system(
    ${CIB_PREFIX}_msg
    INTERFACE
    async
    ${CIB_PREFIX}_log
    ${CIB_PREFIX}_lookup
    ${CIB_PREFIX}_match
    stdx)

target_sources(
    ${CIB_PREFIX}_msg
    INTERFACE FILE_SET
              ${CIB_PREFIX}_msg
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/msg/callback.hpp
              include/msg/detail/indexed_builder_common.hpp
              include/msg/detail/indexed_handler_common.hpp
              include/msg/detail/separate_sum_terms.hpp
              include/msg/field.hpp
              include/msg/field_matchers.hpp
              include/msg/handler_builder.hpp
              include/msg/handler.hpp
              include/msg/handler_interface.hpp
              include/msg/indexed_builder.hpp
              include/msg/indexed_handler.hpp
              include/msg/indexed_service.hpp
              include/msg/message.hpp
              include/msg/send.hpp
              include/msg/service.hpp)

if(PROJECT_IS_TOP_LEVEL)
    clang_tidy_interface(${CIB_PREFIX}_msg)
    add_subdirectory(test)
    add_subdirectory(benchmark)
endif()
