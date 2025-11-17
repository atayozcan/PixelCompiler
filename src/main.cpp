//===- main.cpp - Pixel Compiler Driver ---------------------------------===//
//
// Main driver for the Pixel image processing DSL compiler
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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "pixel_frontend.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
using namespace mlir;
using namespace llvm;
using namespace pixel;
using namespace std;
using namespace cl;

static opt<string> inputFilename(Positional, desc("<input .px file>"),
                                 init("-"), value_desc("filename"));

static opt<bool> dumpHighLevelMLIR("dump-high-level",
                                   desc("Dump high-level Pixel MLIR"),
                                   init(false));

static opt<bool> dumpLowLevelMLIR("dump-low-level",
                                  desc("Dump lowered standard MLIR"),
                                  init(false));

static opt<bool> dumpLLVM("dump-llvm", desc("Dump LLVM IR"), init(false));

static opt<bool> dumpAll("dump-all",
                         desc("Dump all intermediate representations"),
                         init(false));

/// Load a .px script file into a string
static string loadFile(const string &filename) {
  ifstream file(filename);
  if (!file.is_open()) {
    errs() << "Error: Cannot open file '" << filename << "'\n";
    exit(EXIT_FAILURE);
  }
  string content((istreambuf_iterator<char>(file)),
                 istreambuf_iterator<char>());
  return content;
}

/// Run complete lowering pipeline from Pixel to LLVM dialect
LogicalResult lowerToLLVMDialect(ModuleOp module) {
  PassManager pm(module.getContext());

  // Lower Pixel dialect to LLVM dialect with runtime calls
  pm.addPass(createPixelToLLVMLoweringPass());
  // Convert Arith dialect operations to LLVM
  pm.addPass(createArithToLLVMConversionPass());
  // Convert remaining Func dialect operations to LLVM
  pm.addPass(createConvertFuncToLLVMPass());
  // Reconcile unrealized casts
  pm.addPass(createReconcileUnrealizedCastsPass());
  // Canonicalization
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(module)))
    return failure();
  return success();
}

int main(const int argc, char **argv) {
  // Register MLIR command line options
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  ParseCommandLineOptions(argc, argv, "Pixel compiler\n");

  // Initialize LLVM
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  // Create MLIR context
  DialectRegistry registry;
  registry.insert<PixelDialect, func::FuncDialect, arith::ArithDialect,
                  memref::MemRefDialect, LLVM::LLVMDialect>();

  // Register LLVM IR translation for dialects we use
  registerLLVMDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Read input file
  const auto scriptText = loadFile(inputFilename);

  // Parse .px script into high-level MLIR
  outs() << "Parsing Pixel script...\n";
  auto module = parsePixelScript(scriptText, context);

  if (!module) {
    errs() << "Error: Failed to parse Pixel script\n";
    return EXIT_FAILURE;
  }

  // Verify the module
  if (failed(verify(*module))) {
    errs() << "Error: Module verification failed\n";
    module->dump();
    return EXIT_FAILURE;
  }

  // Dump high-level MLIR (Pixel dialect)
  if (dumpHighLevelMLIR || dumpAll) {
    outs() << "\n=== High-Level MLIR (Pixel Dialect) ===\n";
    module->print(outs());
    outs() << "\n";
  }

  // Lower to LLVM dialect
  outs() << "Lowering to LLVM dialect...\n";
  if (failed(lowerToLLVMDialect(*module))) {
    errs() << "Error: Failed to lower to LLVM dialect\n";
    return EXIT_FAILURE;
  }

  // Dump low-level MLIR (after lowering to LLVM dialect)
  if (dumpLowLevelMLIR || dumpAll) {
    outs() << "\n=== Low-Level MLIR (LLVM Dialect) ===\n";
    module->print(outs());
    outs() << "\n";
  }

  // Convert to LLVM IR
  LLVMContext llvmContext;
  const auto llvmModule = translateModuleToLLVMIR(*module, llvmContext);

  if (!llvmModule) {
    errs() << "Error: Failed to convert to LLVM IR\n";
    return EXIT_FAILURE;
  }

  // Dump LLVM IR
  if (dumpLLVM || dumpAll) {
    outs() << "\n=== LLVM IR ===\n";
    llvmModule->print(outs(), nullptr);
    outs() << "\n";
  }

  outs() << "\nCompilation pipeline completed successfully!\n";
  outs() << "Generated LLVM IR can be compiled to an executable using:\n";
  outs() << "  llc output.ll -o output.s\n";
  outs() << "  clang output.s -L. -lPixelRuntime -o pixel_program\n";
  return EXIT_SUCCESS;
}
