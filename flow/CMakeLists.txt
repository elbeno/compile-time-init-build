add_library(${CIB_PREFIX}_flow INTERFACE)
target_compile_features(${CIB_PREFIX}_flow INTERFACE cxx_std_20)
target_link_libraries_system(${CIB_PREFIX}_flow INTERFACE ${CIB_PREFIX}_log
                             ${CIB_PREFIX}_nexus stdx)

target_sources(
    ${CIB_PREFIX}_flow
    INTERFACE FILE_SET
              ${CIB_PREFIX}_flow
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/flow/builder.hpp
              include/flow/debug_builder.hpp
              include/flow/dsl/par.hpp
              include/flow/dsl/seq.hpp
              include/flow/dsl/subgraph_identity.hpp
              include/flow/dsl/walk.hpp
              include/flow/flow.hpp
              include/flow/func_list.hpp
              include/flow/graph_builder.hpp
              include/flow/graph_common.hpp
              include/flow/log.hpp
              include/flow/run.hpp
              include/flow/service.hpp
              include/flow/step.hpp
              include/flow/viz_builder.hpp)

if(PROJECT_IS_TOP_LEVEL)
    clang_tidy_interface(${CIB_PREFIX}_flow)
    add_subdirectory(test)
endif()
