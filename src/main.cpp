//===- main.cpp - Pixel Compiler -----------------------------------------===//
//
// Simple compiler for Pixel image processing DSL
//
//===----------------------------------------------------------------------===//
#include "main.h"

optional<string> loadFile(const string &filename) {
  ifstream file(filename);
  if (!file.is_open()) {
    errs() << "Error: Cannot open file '" << filename << "'\n";
    return nullopt;
  }
  return {{istreambuf_iterator<char>(file), istreambuf_iterator<char>()}};
}

bool writeToFile(const string &filename, const string &content) {
  if (error_code e; !filesystem::create_directory(DIR_OUT, e) && e) {
    errs() << "Error: Could not create directory '" << DIR_OUT
           << "': " << e.message() << "\n";
    return false;
  }

  const auto file_path = DIR_OUT + filename;
  if (ofstream file(file_path); file.is_open()) {
    file << content;
    return true;
  }

  errs() << "Error: Cannot write to file '" << file_path << "'\n";
  return false;
}

bool gen_hi_mlir(const OwningOpRef<ModuleOp> &module) {
  outs() << "[2/9] Generating main program high-level MLIR...\n";
  string highLevelMLIR;
  raw_string_ostream highLevelStream(highLevelMLIR);
  module.get().print(highLevelStream);
  return writeToFile("main_high_level.mlir", highLevelMLIR);
}

bool gen_lo_mlir(const OwningOpRef<ModuleOp> &module) {
  outs() << "[4/9] Generating main program low-level MLIR...\n";
  string lowLevelMLIR;
  raw_string_ostream lowLevelStream(lowLevelMLIR);
  module.get().print(lowLevelStream);
  return writeToFile("main_low_level.mlir", lowLevelMLIR);
}

bool gen_llvm() {
  outs() << "[5/9] Generating main program LLVM IR...\n";
  if (system(CMD_GEN_LLVM.c_str()) == EXIT_SUCCESS)
    return true;
  errs() << "Error: Failed to generate LLVM IR (mlir-translate not found)\n";
  return false;
}

void gen_asm() {
  outs() << "[6/9] Generating main program assembly...\n";
  if (system(CMD_GEN_ASM.c_str()) == EXIT_SUCCESS)
    return;
  errs() << "Warning: Failed to generate assembly for main program\n";
}

bool gen_runtime_llvm() {
  outs() << "[7/9] Generating runtime LLVM IR...\n";
  if (system(CMD_GEN_RUNTIME_LLVM.c_str()) == EXIT_SUCCESS)
    return true;
  errs() << "Error: Failed to generate runtime LLVM IR\n";
  return false;
}

void gen_runtime_mlir() {
  outs() << "[8/9] Generating runtime MLIR...\n";
  if (system(CMD_GEN_RUNTIME_MLIR.c_str()) == EXIT_SUCCESS)
    return;
  errs() << "Warning: Failed to generate runtime MLIR\n";
}

void gen_runtime_asm() {
  if (system(CMD_GEN_RUNTIME_ASM.c_str()) == EXIT_SUCCESS)
    return;
  errs() << "Warning: Failed to generate runtime assembly\n";
}

bool gen_obj() {
  if (system(GEN_OBJ.c_str()) == EXIT_SUCCESS)
    return true;
  errs() << "Error: Failed to compile runtime\n";
  return false;
}

bool link_exec() {
  outs() << "[9/9] Linking executable...\n";
  string outputName = filesystem::path(inputFile).stem().string();
  const string cmd =
      format("clang++ {0}main.ll {0}pixel_runtime.o -o {0}{1} -lm 2>/dev/null",
             DIR_OUT, outputName);
  if (system(cmd.c_str()) == EXIT_SUCCESS)
    return true;
  errs() << "Error: Failed to link executable\n";
  return false;
}

int main(const int argc, char **argv) {
  if (argc != 2) {
    errs() << "Usage: " << argv[0] << " <input.px>\n";
    return EXIT_FAILURE;
  }
  inputFile = argv[1];

  // Create MLIR context with all necessary dialects
  MLIRContext context;
  context.getOrLoadDialect<PixelDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();

  outs() << "Compiling Pixel Program\n";
  // ========================================================================
  // PART 1: Compile Main Program (Pixel Script)
  // ========================================================================
  // Parse .px script
  outs() << "[1/9] Parsing Pixel script...\n";
  const auto fileContent = loadFile(inputFile);
  if (!fileContent) return EXIT_FAILURE;

  auto module = parsePixelScript(*fileContent, context);

  if (!module || failed(verify(*module))) {
    errs() << "Error: Failed to parse or verify script\n";
    return EXIT_FAILURE;
  }

  // Save high-level MLIR (Pixel dialect)
  if (!gen_hi_mlir(module))
    return EXIT_FAILURE;

  // Lower to LLVM dialect
  outs() << "[3/9] Lowering main program to LLVM dialect...\n";
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

  // Save low-level MLIR (LLVM dialect)
  if (!gen_lo_mlir(module))
    return EXIT_FAILURE;

  // Generate LLVM IR for main program
  if (!gen_llvm()) return EXIT_FAILURE;

  // Generate assembly for main program
  gen_asm();

  // ========================================================================
  // PART 2: Compile Runtime Library
  // ========================================================================
  // Generate runtime LLVM IR
  if (!gen_runtime_llvm()) return EXIT_FAILURE;

  // Convert runtime LLVM IR to MLIR
  gen_runtime_mlir();

  // Generate runtime assembly
  gen_runtime_asm();

  // Compile runtime to object file
  if (!gen_obj())
    return EXIT_FAILURE;

  // ========================================================================
  // PART 3: Link Final Executable
  // ========================================================================
  if (!link_exec())
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
