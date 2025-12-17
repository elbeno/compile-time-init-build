add_library(${CIB_PREFIX}_match INTERFACE)
target_compile_features(${CIB_PREFIX}_match INTERFACE cxx_std_20)
target_link_libraries_system(${CIB_PREFIX}_match INTERFACE stdx)

target_sources(
    ${CIB_PREFIX}_match
    INTERFACE FILE_SET
              ${CIB_PREFIX}_match
              TYPE
              HEADERS
              BASE_DIRS
              include
              FILES
              include/match/and.hpp
              include/match/bin_op.hpp
              include/match/concepts.hpp
              include/match/constant.hpp
              include/match/cost.hpp
              include/match/implies.hpp
              include/match/negate.hpp
              include/match/not.hpp
              include/match/ops.hpp
              include/match/or.hpp
              include/match/predicate.hpp
              include/match/simplify.hpp
              include/match/sum_of_products.hpp)

if(PROJECT_IS_TOP_LEVEL)
    clang_tidy_interface(${CIB_PREFIX}_match)
    add_subdirectory(test)
endif()
