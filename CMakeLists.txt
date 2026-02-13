cmake_minimum_required(VERSION 4.2.3)

project(tableau VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only, generic integrators.
add_library(tableau INTERFACE)

target_include_directories(tableau
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(tableau INTERFACE cxx_std_17)

if(MSVC)
    target_compile_options(tableau INTERFACE /W4)
else()
    target_compile_options(tableau INTERFACE -Wall -Wextra -pedantic)
endif()

# Create alias for easier use
add_library(tableau::core ALIAS tableau)

# No install rules needed as it's an INTERFACE target.

include(CTest)
option(BUILD_DEMOS "Build optional interactive demos" OFF)

if(BUILD_TESTING)
    add_executable(tableau_tests
        tests/test_dop853.cpp
    )
    target_link_libraries(tableau_tests PRIVATE tableau)
    target_compile_features(tableau_tests PRIVATE cxx_std_17)
    add_test(NAME tableau_tests COMMAND tableau_tests)

    add_executable(rk4_tests
        tests/test_rk4.cpp
    )
    target_link_libraries(rk4_tests PRIVATE tableau)
    target_compile_features(rk4_tests PRIVATE cxx_std_17)
    add_test(NAME rk4_tests COMMAND rk4_tests)

    add_executable(rkf45_tests
        tests/test_rkf45.cpp
    )
    target_link_libraries(rkf45_tests PRIVATE tableau)
    target_compile_features(rkf45_tests PRIVATE cxx_std_17)
    add_test(NAME rkf45_tests COMMAND rkf45_tests)

    add_executable(dop853_extended_tests
        tests/test_dop853_extended.cpp
    )
    target_link_libraries(dop853_extended_tests PRIVATE tableau)
    target_compile_features(dop853_extended_tests PRIVATE cxx_std_17)
    add_test(NAME dop853_extended_tests COMMAND dop853_extended_tests)

    add_executable(three_body_core_tests
        tests/test_three_body_core.cpp
    )
    target_link_libraries(three_body_core_tests PRIVATE tableau)
    target_include_directories(three_body_core_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/demos/src
    )
    target_compile_features(three_body_core_tests PRIVATE cxx_std_17)
    add_test(NAME three_body_core_tests COMMAND three_body_core_tests)

    add_executable(black_hole_core_tests
        tests/test_black_hole_core.cpp
    )
    target_link_libraries(black_hole_core_tests PRIVATE tableau)
    target_include_directories(black_hole_core_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/demos/src
    )
    target_compile_features(black_hole_core_tests PRIVATE cxx_std_17)
    add_test(NAME black_hole_core_tests COMMAND black_hole_core_tests)

    add_executable(golf_core_tests
        tests/test_golf_core.cpp
    )
    target_link_libraries(golf_core_tests PRIVATE tableau)
    target_include_directories(golf_core_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/demos/src
    )
    target_compile_features(golf_core_tests PRIVATE cxx_std_17)
    add_test(NAME golf_core_tests COMMAND golf_core_tests)
endif()

option(BUILD_BENCHMARKS "Build Google Benchmark suite (requires local google-benchmark)" OFF)

if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)

    add_executable(tableau_benchmarks
        benchmarks/bench_integrators.cpp
    )
    target_link_libraries(tableau_benchmarks
        PRIVATE
            tableau
            benchmark::benchmark
    )
    target_compile_features(tableau_benchmarks PRIVATE cxx_std_17)
endif()

if(BUILD_DEMOS)
    add_subdirectory(demos)
endif()
