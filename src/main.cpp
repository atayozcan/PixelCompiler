//===- main.cpp - Pixel Compiler -----------------------------------------===//
//
// Simple compiler for Pixel image processing DSL
//
//===----------------------------------------------------------------------===//
#include "PixelDialect.h"
#include "PixelPasses.h"
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
#include "../include/pixel_frontend.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
using namespace mlir;
using namespace pixel;
using namespace std;
using llvm::outs;
using llvm::errs;
using llvm::raw_string_ostream;

string loadFile(const string &filename) {
  ifstream file(filename);
  if (!file.is_open()) {
    errs() << "Error: Cannot open file '" << filename << "'\n";
    exit(EXIT_FAILURE);
  }
  return {(istreambuf_iterator<char>(file)), istreambuf_iterator<char>()};
}

void writeToFile(const string &filename, const string &content) {
  if (ofstream file(filename); file.is_open()) {
    file << content;
    return;
  }
  errs() << "Error: Cannot write to file '" << filename << "'\n";
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    errs() << "Usage: " << argv[0] << " <input.px>\n";
    return 1;
  }

  auto inputFile = argv[1];

  // Create MLIR context with all necessary dialects
  MLIRContext context;
  context.getOrLoadDialect<PixelDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();

  // Parse .px script
  outs() << "[1/5] Parsing Pixel script...\n";
  auto module = parsePixelScript(loadFile(inputFile), context);

  if (!module || failed(verify(*module))) {
    errs() << "Error: Failed to parse or verify script\n";
    return EXIT_FAILURE;
  }

  // Save high-level MLIR
  outs() << "[2/5] Generating high-level MLIR...\n";
  string highLevelMLIR;
  raw_string_ostream highLevelStream(highLevelMLIR);
  module->print(highLevelStream);
  writeToFile("output_high_level.mlir", highLevelMLIR);

  // Lower to LLVM dialect
  outs() << "[3/5] Lowering to LLVM dialect...\n";
  PassManager pm(module->getContext());
  pm.addPass(createPixelToLLVMLoweringPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(*module))) {
    errs() << "Error: Failed to lower to LLVM dialect\n";
    return EXIT_FAILURE;
  }

  // Save low-level MLIR
  string lowLevelMLIR;
  raw_string_ostream lowLevelStream(lowLevelMLIR);
  module->print(lowLevelStream);
  writeToFile("output_low_level.mlir", lowLevelMLIR);

  // Generate LLVM IR
  outs() << "[4/5] Generating LLVM IR...\n";
  int result = system(
    "command -v mlir-translate >/dev/null 2>&1 && mlir-translate --mlir-to-llvmir output_low_level.mlir -o output.ll 2>/dev/null || "
    "../llvm/bin/mlir-translate --mlir-to-llvmir output_low_level.mlir -o output.ll 2>/dev/null || "
    "llvm/bin/mlir-translate --mlir-to-llvmir output_low_level.mlir -o output.ll 2>/dev/null");

  if (result != EXIT_SUCCESS) {
    errs() << "Error: Failed to generate LLVM IR (mlir-translate not found)\n";
    return EXIT_FAILURE;
  }

  // Compile to executable
  outs() << "[5/5] Compiling to executable...\n";
  result = system("clang++ -c src/pixel_runtime.cpp -Iinclude -o pixel_runtime.o 2>/dev/null && "
    "clang++ output.ll pixel_runtime.o -o pixel_program -lm 2>/dev/null");

  if (result != EXIT_SUCCESS) {
    errs() << "Error: Failed to compile executable\n";
    return EXIT_FAILURE;
  }

  outs() << "\nSuccess! Generated:\n";
  outs() << "  output_high_level.mlir - High-level Pixel dialect MLIR\n";
  outs() << "  output_low_level.mlir  - Low-level LLVM dialect MLIR\n";
  outs() << "  output.ll              - LLVM IR\n";
  outs() << "  pixel_program          - Executable\n\n";
  outs() << "Run with: ./pixel_program\n";

  return EXIT_SUCCESS;
}
