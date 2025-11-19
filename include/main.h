#pragma once

#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>

#include "PixelDialect.h"
#include "PixelPasses.h"
#include "pixel_frontend.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace pixel;
using namespace std;
using llvm::errs;
using llvm::outs;
using llvm::raw_string_ostream;

constexpr auto DIR_OUT = "out/";
inline char *inputFile = nullptr;

const auto CMD_GEN_LLVM = format(
    "command -v mlir-translate >/dev/null 2>&1 && mlir-translate "
    "--mlir-to-llvmir {0}main_low_level.mlir -o {0}main.ll 2>/dev/null || "
    "include/llvm/bin/mlir-translate --mlir-to-llvmir {0}main_low_level.mlir "
    "-o {0}main.ll 2>/dev/null || llvm/bin/mlir-translate --mlir-to-llvmir "
    "{0}main_low_level.mlir -o {0}main.ll 2>/dev/null",
    DIR_OUT);
const auto CMD_GEN_ASM =
    format("llc {0}main.ll -o {0}main.s 2>/dev/null || "
           "llvm/bin/llc {0}main.ll -o {0}main.s 2>/dev/null",
           DIR_OUT);
const auto CMD_GEN_RUNTIME_LLVM =
    format("clang++ -S -emit-llvm src/pixel_runtime.cpp -Iinclude -o "
           "{0}pixel_runtime.ll 2>/dev/null",
           DIR_OUT);
const auto CMD_GEN_RUNTIME_MLIR = format(
    "command -v mlir-translate >/dev/null 2>&1 && mlir-translate "
    "--import-llvm {0}pixel_runtime.ll -o {0}pixel_runtime.mlir 2>/dev/null || "
    "include/llvm/bin/mlir-translate --import-llvm {0}pixel_runtime.ll -o "
    "{0}pixel_runtime.mlir 2>/dev/null || llvm/bin/mlir-translate "
    "--import-llvm "
    "{0}pixel_runtime.ll -o {0}pixel_runtime.mlir 2>/dev/null",
    DIR_OUT);
const auto CMD_GEN_RUNTIME_ASM =
    format("clang++ -S src/pixel_runtime.cpp -Iinclude -o "
           "{0}pixel_runtime.s 2>/dev/null",
           DIR_OUT);
const auto GEN_OBJ = format("clang++ -c src/pixel_runtime.cpp -Iinclude -o "
                            "{0}pixel_runtime.o 2>/dev/null",
                            DIR_OUT);
