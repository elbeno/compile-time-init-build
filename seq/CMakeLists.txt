add_library(${CIB_PREFIX}_seq INTERFACE)
target_compile_features(${CIB_PREFIX}_seq INTERFACE cxx_std_20)
target_link_libraries_system(${CIB_PREFIX}_seq INTERFACE ${CIB_PREFIX}_flow
                             ${CIB_PREFIX}_log ${CIB_PREFIX}_nexus stdx)

target_sources(
    ${CIB_PREFIX}_seq
    INTERFACE FILE_SET
              ${CIB_PREFIX}_seq
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/seq/builder.hpp
              include/seq/impl.hpp
              include/seq/step.hpp)

if(PROJECT_IS_TOP_LEVEL)
    clang_tidy_interface(${CIB_PREFIX}_seq)
    add_subdirectory(test)
endif()
