add_library(${CIB_PREFIX}_nexus INTERFACE)
target_compile_features(${CIB_PREFIX}_nexus INTERFACE cxx_std_20)
target_link_libraries_system(${CIB_PREFIX}_nexus INTERFACE stdx)

target_sources(
    ${CIB_PREFIX}_nexus
    INTERFACE FILE_SET
              ${CIB_PREFIX}_nexus
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/${CIB_PREFIX}/builder_meta.hpp
              include/${CIB_PREFIX}/built.hpp
              include/${CIB_PREFIX}/callback.hpp
              include/${CIB_PREFIX}/config.hpp
              include/${CIB_PREFIX}/detail/components.hpp
              include/${CIB_PREFIX}/detail/constexpr_conditional.hpp
              include/${CIB_PREFIX}/detail/runtime_conditional.hpp
              include/${CIB_PREFIX}/detail/config_details.hpp
              include/${CIB_PREFIX}/detail/config_item.hpp
              include/${CIB_PREFIX}/detail/exports.hpp
              include/${CIB_PREFIX}/detail/extend.hpp
              include/${CIB_PREFIX}/detail/nexus_details.hpp
              include/${CIB_PREFIX}/func_decl.hpp
              include/${CIB_PREFIX}/nexus.hpp)

if(PROJECT_IS_TOP_LEVEL)
    clang_tidy_interface(${CIB_PREFIX}_nexus)
    add_subdirectory(test)
    add_subdirectory(benchmark)
endif()
