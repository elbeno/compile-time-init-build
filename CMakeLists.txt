cmake_minimum_required(VERSION 3.25)

project(
    cib
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION
        "A header-only C++ library for composing modular firmware at compile-time."
    HOMEPAGE_URL "https://github.com/intel/compile-time-init-build")

include(cmake/get_cpm.cmake)
if(PROJECT_IS_TOP_LEVEL)
    cpmaddpackage("gh:intel/cicd-repo-infrastructure#dev")
else()
    cpmaddpackage("gh:intel/cicd-repo-infrastructure#3e2bef0")
endif()

find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)

include(cmake/debug_flow.cmake)
include(cmake/string_catalog.cmake)

add_versioned_package("gh:boostorg/mp11#boost-1.83.0")
fmt_recipe(11.1.3)
add_versioned_package("gh:intel/cpp-baremetal-concurrency#de60a38")
add_versioned_package("gh:intel/cpp-std-extensions#a1ae0b4")
add_versioned_package("gh:intel/cpp-baremetal-senders-and-receivers#27db6e1")

set(GEN_STR_CATALOG
    ${CMAKE_CURRENT_LIST_DIR}/tools/gen_str_catalog.py
    CACHE FILEPATH "Location of script to generate a string catalog")

if(NOT DEFINED CIB_PREFIX)
    set(CIB_PREFIX cib)
endif()

function(add_tests)
    set(singleValueArgs PREFIX)
    set(multiValueArgs FILES LIBRARIES)
    cmake_parse_arguments(AT "" "${singleValueArgs}" "${multiValueArgs}"
                          ${ARGN})

    foreach(name ${AT_FILES})
        string(REPLACE "/" "_" test ${name})
        add_unit_test(
            "${CIB_PREFIX}_${AT_PREFIX}_${test}_test"
            CATCH2
            FILES
            "${name}.cpp"
            LIBRARIES
            warnings
            ${AT_LIBRARIES})
    endforeach()
endfunction()

add_subdirectory(interrupt)
add_subdirectory(log)
add_subdirectory(lookup)
add_subdirectory(match)
add_subdirectory(nexus)

add_subdirectory(flow)
add_subdirectory(msg)
add_subdirectory(seq)

add_library(cib INTERFACE)
target_compile_features(cib INTERFACE cxx_std_20)
target_link_libraries_system(
    cib
    INTERFACE
    cib_flow
    cib_interrupt
    cib_log
    cib_log_binary
    cib_log_fmt
    cib_lookup
    cib_match
    cib_msg
    cib_nexus
    cib_seq
    stdx)

target_sources(
    cib
    INTERFACE FILE_SET
              cib
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/cib/cib.hpp
              include/cib/top.hpp)

if(PROJECT_IS_TOP_LEVEL)
    add_docs(docs)
    clang_tidy_interface(cib)

    # Enable functional and performance test suites.
    add_subdirectory(examples)
    add_subdirectory(tools)

    # Build single-header release.
    include(cmake/single_header.cmake)
    gen_single_header(
        TARGET
        release_header
        GEN_HEADER
        ${CMAKE_SOURCE_DIR}/tools/gen_release_header.py
        SOURCE_TARGET
        cib
        SOURCE_FILESET
        cib
        INPUT_HEADER
        ${CMAKE_SOURCE_DIR}/include/cib/cib.hpp
        OUTPUT_HEADER
        ${CMAKE_BINARY_DIR}/include/cib/cib.hpp)

    if(DEFINED ENV{SINGLE_HEADER})
        add_dependencies(cib release_header)

        target_include_directories(
            cib
            INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)
    else()
        target_include_directories(
            cib
            INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)
    endif()
endif()
