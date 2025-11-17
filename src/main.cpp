//===- main.cpp - Pixel Compiler Driver ---------------------------------===//
//
// Main driver for the Pixel image processing DSL compiler
//
//===----------------------------------------------------------------------===//

#include "PixelDialect.h"
#include "PixelPasses.h"
#include "pixel_frontend.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// ExecutionEngine removed for compatibility
// #include "mlir/ExecutionEngine/ExecutionEngine.h"
// #include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace mlir;

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .px file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> dumpHighLevelMLIR("dump-high-level",
                                        cl::desc("Dump high-level Pixel MLIR"),
                                        cl::init(false));

static cl::opt<bool> dumpLowLevelMLIR("dump-low-level",
                                       cl::desc("Dump lowered standard MLIR"),
                                       cl::init(false));

static cl::opt<bool> dumpLLVM("dump-llvm",
                               cl::desc("Dump LLVM IR"),
                               cl::init(false));

static cl::opt<bool> dumpAll("dump-all",
                              cl::desc("Dump all intermediate representations"),
                              cl::init(false));

// JIT execution disabled for compatibility
// static cl::opt<bool> runJIT("run",
//                              cl::desc("JIT compile and run the code"),
//                              cl::init(false));

/// Load a .px script file into a string
static std::string loadFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    llvm::errs() << "Error: Cannot open file '" << filename << "'\n";
    exit(1);
  }
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  return content;
}

/// Run complete lowering pipeline from Pixel to LLVM dialect
LogicalResult lowerToLLVMDialect(ModuleOp module) {
  PassManager pm(module.getContext());

  // Lower Pixel dialect to LLVM dialect with runtime calls
  pm.addPass(pixel::createPixelToLLVMLoweringPass());

  // Convert remaining Func dialect operations to LLVM
  pm.addPass(createConvertFuncToLLVMPass());

  // Reconcile unrealized casts
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Canonicalization
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(module))) {
    return failure();
  }

  return success();
}

int main(int argc, char **argv) {
  // Register MLIR command line options
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "Pixel compiler\n");

  // Initialize LLVM
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create MLIR context
  DialectRegistry registry;
  registry.insert<pixel::PixelDialect, func::FuncDialect,
                  arith::ArithDialect, memref::MemRefDialect,
                  LLVM::LLVMDialect>();

  // Register LLVM IR translation for dialects we use
  registerLLVMDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Read input file
  std::string scriptText = loadFile(inputFilename);

  // Parse .px script into high-level MLIR
  llvm::outs() << "Parsing Pixel script...\n";
  auto module = parsePixelScript(scriptText, context);

  if (!module) {
    llvm::errs() << "Error: Failed to parse Pixel script\n";
    return 1;
  }

  // Verify the module
  if (failed(verify(*module))) {
    llvm::errs() << "Error: Module verification failed\n";
    module->dump();
    return 1;
  }

  // Dump high-level MLIR (Pixel dialect)
  if (dumpHighLevelMLIR || dumpAll) {
    llvm::outs() << "\n=== High-Level MLIR (Pixel Dialect) ===\n";
    module->print(llvm::outs());
    llvm::outs() << "\n";
  }

  // Clone module for low-level MLIR dump
  auto clonedModule = module->clone();

  // Lower to LLVM dialect
  llvm::outs() << "Lowering to LLVM dialect...\n";
  if (failed(lowerToLLVMDialect(*module))) {
    llvm::errs() << "Error: Failed to lower to LLVM dialect\n";
    return 1;
  }

  // Dump low-level MLIR (after lowering to LLVM dialect)
  if (dumpLowLevelMLIR || dumpAll) {
    llvm::outs() << "\n=== Low-Level MLIR (LLVM Dialect) ===\n";
    module->print(llvm::outs());
    llvm::outs() << "\n";
  }

  // Convert to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(*module, llvmContext);

  if (!llvmModule) {
    llvm::errs() << "Error: Failed to convert to LLVM IR\n";
    return 1;
  }

  // Dump LLVM IR
  if (dumpLLVM || dumpAll) {
    llvm::outs() << "\n=== LLVM IR ===\n";
    llvmModule->print(llvm::outs(), nullptr);
    llvm::outs() << "\n";
  }

  llvm::outs() << "\nCompilation pipeline completed successfully!\n";
  llvm::outs() << "Generated LLVM IR can be compiled to an executable using:\n";
  llvm::outs() << "  llc output.ll -o output.s\n";
  llvm::outs() << "  clang output.s -L. -lPixelRuntime -o pixel_program\n";
  return 0;
}
