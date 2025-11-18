cmake_minimum_required(VERSION 4.0)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

project(PixelCompiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld -stdlib=libc++ -lc++abi")

# Find zstd
find_library(ZSTD_LIB zstd REQUIRED)
add_library(zstd::libzstd_static UNKNOWN IMPORTED)
set_target_properties(zstd::libzstd_static PROPERTIES IMPORTED_LOCATION "${ZSTD_LIB}")

# Find LLVM and MLIR
set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm/lib/cmake/llvm")
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm/lib/cmake/mlir")
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})

# Generate dialect files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)
set(LLVM_TARGET_DEFINITIONS include/PixelOps.td)
mlir_tablegen(include/PixelOps.h.inc -gen-op-decls)
mlir_tablegen(include/PixelOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/PixelOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/PixelOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(PixelOpsIncGen)

# Pixel dialect library
add_library(PixelDialect STATIC src/PixelCompiler.cpp)
add_dependencies(PixelDialect PixelOpsIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(PixelDialect PUBLIC
        ${dialect_libs} ${conversion_libs}
        MLIRIR MLIRParser MLIRPass MLIRTransforms
        MLIRLLVMCommonConversion MLIRFuncToLLVM MLIRSupport)

# Compiler executable
add_executable(pixel_compiler src/main.cpp src/pixel_frontend.cpp)

llvm_map_components_to_libnames(llvm_libs
        Support Core
        X86CodeGen X86AsmParser X86Desc X86Info)

target_link_libraries(pixel_compiler PRIVATE
        PixelDialect
        -Wl,--whole-archive ${dialect_libs} ${conversion_libs} -Wl,--no-whole-archive
        MLIRIR MLIRParser MLIRPass MLIRTransforms
        MLIRTargetLLVMIRExport MLIRLLVMToLLVMIRTranslation
        MLIRLLVMCommonConversion MLIRFuncToLLVM MLIRSupport
        ${llvm_libs} zstd pthread m dl)
