cmake_minimum_required(VERSION 4.1)

# Use clang compiler (must be set before project())
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

project(PixelCompiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Match LLVM's configuration: RTTI, libc++, and lld for LTO bitcode
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld -stdlib=libc++ -lc++abi")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld -stdlib=libc++")

find_library(ZSTD_LIB zstd REQUIRED)
add_library(zstd::libzstd_static UNKNOWN IMPORTED)
set_target_properties(zstd::libzstd_static PROPERTIES IMPORTED_LOCATION "${ZSTD_LIB}")

#===----------------------------------------------------------------------===//
# Find LLVM and MLIR
#===----------------------------------------------------------------------===//

# Use downloaded LLVM 21 prebuilt binaries
set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm/lib/cmake/llvm")
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm/lib/cmake/mlir")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#===----------------------------------------------------------------------===//
# TableGen for Pixel Dialect
#===----------------------------------------------------------------------===//

# Create the output directory for generated files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

set(LLVM_TARGET_DEFINITIONS include/PixelOps.td)
mlir_tablegen(include/PixelOps.h.inc -gen-op-decls)
mlir_tablegen(include/PixelOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/PixelOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/PixelOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(PixelOpsIncGen)

#===----------------------------------------------------------------------===//
# Pixel Dialect Library
#===----------------------------------------------------------------------===//

add_library(PixelDialect STATIC
        src/PixelDialect.cpp
        src/PixelToLLVMLowering.cpp
)

add_dependencies(PixelDialect PixelOpsIncGen)

target_include_directories(PixelDialect PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

get_property(dialect_libs_for_pixel GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs_for_pixel GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(PixelDialect
        PUBLIC
        ${dialect_libs_for_pixel}
        ${conversion_libs_for_pixel}
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTransforms
        MLIRLLVMCommonConversion
        MLIRFuncToLLVM
        MLIRSupport
)

#===----------------------------------------------------------------------===//
# Pixel Compiler Executable
#===----------------------------------------------------------------------===//

add_executable(pixel_compiler
        src/main.cpp
        src/pixel_frontend.cpp
)

llvm_map_components_to_libnames(llvm_libs
        Support
        Core
        X86CodeGen
        X86AsmParser
        X86Desc
        X86Info
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(pixel_compiler
        PRIVATE
        PixelDialect
        PixelRuntime
        -Wl,--whole-archive
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTransforms
        -Wl,--no-whole-archive
        MLIRExecutionEngine
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        MLIRLLVMCommonConversion
        MLIRFuncToLLVM
        MLIRSupport
        MLIRSideEffectInterfaces
        ${llvm_libs}
        zstd::libzstd_static
        pthread
        m
        dl
)

#===----------------------------------------------------------------------===//
# Pixel Runtime Library - for actual BMP image processing
#===----------------------------------------------------------------------===//

add_library(PixelRuntime SHARED
        src/pixel_runtime.cpp
)

target_include_directories(PixelRuntime PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

#===----------------------------------------------------------------------===//
# Installation
#===----------------------------------------------------------------------===//

install(TARGETS pixel_compiler
        RUNTIME DESTINATION bin
)

install(TARGETS PixelRuntime
        LIBRARY DESTINATION lib
)
