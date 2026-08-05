cmake_minimum_required(VERSION 3.24)

project(
    TransformerToy
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# ============================================================
# Language configuration
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# RTX 40-series desktop and laptop GPUs use compute capability 8.9.
set(CMAKE_CUDA_ARCHITECTURES 89)

# Place generated executables and libraries in predictable folders.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Keep Visual Studio Debug and Release output in the same general layout.
foreach(CONFIG_TYPE Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${CONFIG_TYPE}" CONFIG_TYPE_UPPER)

    set(
        CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE_UPPER}
        "${CMAKE_BINARY_DIR}/bin/${CONFIG_TYPE}"
    )

    set(
        CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPE_UPPER}
        "${CMAKE_BINARY_DIR}/lib/${CONFIG_TYPE}"
    )

    set(
        CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_TYPE_UPPER}
        "${CMAKE_BINARY_DIR}/lib/${CONFIG_TYPE}"
    )
endforeach()

option(
    TRANSFORMER_TOY_BUILD_TESTS
    "Build the native C++ test executable"
    ON
)

option(
    TRANSFORMER_TOY_BUILD_PYTHON
    "Build the Python extension"
    ON
)

# ============================================================
# CUDA toolkit
# ============================================================

find_package(CUDAToolkit REQUIRED)

# ============================================================
# Production source files
# ============================================================

file(
    GLOB_RECURSE TRANSFORMER_TOY_CPP_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

file(
    GLOB_RECURSE TRANSFORMER_TOY_CUDA_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Remove all native test implementation files from the core library.
list(
    FILTER TRANSFORMER_TOY_CPP_SOURCES
    EXCLUDE REGEX "/tests/"
)

list(
    FILTER TRANSFORMER_TOY_CUDA_SOURCES
    EXCLUDE REGEX "/tests/"
)

# Future Python binding files must not become part of the core library.
list(
    FILTER TRANSFORMER_TOY_CPP_SOURCES
    EXCLUDE REGEX "/bindings/"
)

list(
    FILTER TRANSFORMER_TOY_CUDA_SOURCES
    EXCLUDE REGEX "/bindings/"
)

# Exclude known blank placeholder translation units.
list(
    FILTER TRANSFORMER_TOY_CUDA_SOURCES
    EXCLUDE REGEX "reduction_kernels\\.cu$"
)

# ============================================================
# Native core library
# ============================================================

add_library(
    transformer_toy_core
    STATIC
    ${TRANSFORMER_TOY_CPP_SOURCES}
    ${TRANSFORMER_TOY_CUDA_SOURCES}
)

target_include_directories(
    transformer_toy_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(
    transformer_toy_core
    PUBLIC
        CUDA::cudart
)

set_target_properties(
    transformer_toy_core
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Helpful warnings for normal C++ compilation.
if(MSVC)
    target_compile_options(
        transformer_toy_core
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(
        transformer_toy_core
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

# ============================================================
# Native test executable
# ============================================================

file(
    GLOB_RECURSE TRANSFORMER_TOY_TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cu"
)

if(TRANSFORMER_TOY_BUILD_TESTS)
    add_executable(
        transformer_toy_tests
        ${TRANSFORMER_TOY_TEST_SOURCES}
    )

    target_include_directories(
        transformer_toy_tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

    target_link_libraries(
        transformer_toy_tests
        PRIVATE
            transformer_toy_core
    )

    set_target_properties(
        transformer_toy_tests
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ============================================================
# CTest integration
# ============================================================

include(CTest)

if(BUILD_TESTING)
    add_test(
        NAME transformer_toy_native_tests
        COMMAND transformer_toy_tests
    )
endif()

# ============================================================
# Python extension module
# ============================================================

if(TRANSFORMER_TOY_BUILD_PYTHON)
    find_package(
        Python
        COMPONENTS
            Interpreter
            Development.Module
        REQUIRED
    )

    find_package(
        pybind11
        CONFIG
        REQUIRED
    )

    pybind11_add_module(
        _transformer_toy
        MODULE
        "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings.cpp"
    )

    target_link_libraries(
        _transformer_toy
        PRIVATE
            transformer_toy_core
    )

    install(
        TARGETS _transformer_toy
        LIBRARY DESTINATION transformer_toy
        RUNTIME DESTINATION transformer_toy
        ARCHIVE DESTINATION transformer_toy
    )
endif()

target_include_directories(
    _transformer_toy
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set(
    TRANSFORMER_TOY_PYTHON_PACKAGE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/python/transformer_toy"
)

add_custom_command(
    TARGET _transformer_toy
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        "${TRANSFORMER_TOY_PYTHON_PACKAGE_DIR}"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:_transformer_toy>"
        "${TRANSFORMER_TOY_PYTHON_PACKAGE_DIR}/$<TARGET_FILE_NAME:_transformer_toy>"
    COMMENT
        "Copying Python extension into transformer_toy package"
)