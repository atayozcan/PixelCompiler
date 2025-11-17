//===- pixel_frontend.cpp - Parser for Pixel DSL ------------------------===//
//
// Parser implementation for .px files
//
//===----------------------------------------------------------------------===//

#include "pixel_frontend.h"
#include "PixelDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::pixel;

namespace {

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

enum Token {
  tok_eof = -1,
  tok_image = -2,
  tok_operations = -3,
  tok_as = -4,
  tok_bmp = -5,
  tok_identifier = -6,
  tok_string = -7,
  tok_arrow = -8,  // ->
  tok_colon = -9,
  tok_invert = -10,
  tok_grayscale = -11,
  tok_rotate = -12,
  tok_number = -13,
};

class Lexer {
  std::string input;
  size_t pos = 0;

public:
  std::string identifierStr;
  std::string stringVal;
  double numberVal;

  explicit Lexer(std::string input) : input(std::move(input)) {}

  int getNextToken() {
    // Skip whitespace and comments
    while (pos < input.size() && (std::isspace(input[pos]) || input[pos] == '#')) {
      if (input[pos] == '#') {
        // Skip comment line
        while (pos < input.size() && input[pos] != '\n')
          pos++;
      } else {
        pos++;
      }
    }

    if (pos >= input.size())
      return tok_eof;

    // String literals
    if (input[pos] == '"') {
      pos++; // skip opening quote
      stringVal.clear();
      while (pos < input.size() && input[pos] != '"') {
        stringVal += input[pos++];
      }
      if (pos < input.size())
        pos++; // skip closing quote
      return tok_string;
    }

    // Arrow ->
    if (input[pos] == '-' && pos + 1 < input.size() && input[pos + 1] == '>') {
      pos += 2;
      return tok_arrow;
    }

    // Colon
    if (input[pos] == ':') {
      pos++;
      return tok_colon;
    }

    // Numbers
    if (std::isdigit(input[pos]) || input[pos] == '-') {
      std::string numStr;
      numStr += input[pos++];
      while (pos < input.size() && (std::isdigit(input[pos]) || input[pos] == '.')) {
        numStr += input[pos++];
      }
      numberVal = std::stod(numStr);
      return tok_number;
    }

    // Identifiers and keywords
    if (std::isalpha(input[pos]) || input[pos] == '_') {
      identifierStr.clear();
      while (pos < input.size() && (std::isalnum(input[pos]) || input[pos] == '_')) {
        identifierStr += input[pos++];
      }

      if (identifierStr == "image")
        return tok_image;
      if (identifierStr == "operations")
        return tok_operations;
      if (identifierStr == "as")
        return tok_as;
      if (identifierStr == "bmp")
        return tok_bmp;
      if (identifierStr == "invert")
        return tok_invert;
      if (identifierStr == "grayscale")
        return tok_grayscale;
      if (identifierStr == "rotate")
        return tok_rotate;

      return tok_identifier;
    }

    // Unknown character
    pos++;
    return input[pos - 1];
  }
};

//===----------------------------------------------------------------------===//
// AST Nodes
//===----------------------------------------------------------------------===//

struct ImageDecl {
  std::string name;
  std::string inputPath;
  std::string outputPath;
  std::string format;
};

enum OpType {
  OpInvert,
  OpGrayscale,
  OpRotate,
};

struct Operation {
  OpType type;
  double angle = 0.0; // for rotate
};

struct OperationsBlock {
  std::string imageName;
  std::vector<Operation> ops;
};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

class Parser {
  Lexer &lexer;
  int curTok;

  std::vector<ImageDecl> imageDecls;
  std::vector<OperationsBlock> operationsBlocks;

public:
  explicit Parser(Lexer &lexer) : lexer(lexer) {
    getNextToken(); // Prime the lexer
  }

  int getNextToken() { return curTok = lexer.getNextToken(); }

  bool parseScript() {
    while (curTok != tok_eof) {
      if (curTok == tok_image) {
        if (!parseImageDecl())
          return false;
      } else if (curTok == tok_operations) {
        if (!parseOperationsBlock())
          return false;
      } else {
        llvm::errs() << "Unexpected token in script\n";
        return false;
      }
    }
    return true;
  }

  bool parseImageDecl() {
    getNextToken(); // consume 'image'

    if (curTok != tok_identifier) {
      llvm::errs() << "Expected identifier after 'image'\n";
      return false;
    }

    ImageDecl decl;
    decl.name = lexer.identifierStr;
    getNextToken();

    if (curTok != tok_string) {
      llvm::errs() << "Expected input path string\n";
      return false;
    }
    decl.inputPath = lexer.stringVal;
    getNextToken();

    if (curTok != tok_arrow) {
      llvm::errs() << "Expected '->'\n";
      return false;
    }
    getNextToken();

    if (curTok != tok_string) {
      llvm::errs() << "Expected output path string\n";
      return false;
    }
    decl.outputPath = lexer.stringVal;
    getNextToken();

    if (curTok != tok_as) {
      llvm::errs() << "Expected 'as'\n";
      return false;
    }
    getNextToken();

    if (curTok != tok_bmp) {
      llvm::errs() << "Expected format (bmp)\n";
      return false;
    }
    decl.format = "bmp";
    getNextToken();

    imageDecls.push_back(decl);
    return true;
  }

  bool parseOperationsBlock() {
    getNextToken(); // consume 'operations'

    if (curTok != tok_identifier) {
      llvm::errs() << "Expected image name after 'operations'\n";
      return false;
    }

    OperationsBlock block;
    block.imageName = lexer.identifierStr;
    getNextToken();

    if (curTok != tok_colon) {
      llvm::errs() << "Expected ':' after image name\n";
      return false;
    }
    getNextToken();

    // Parse operations
    while (curTok != tok_eof && curTok != tok_image && curTok != tok_operations) {
      if (curTok == tok_invert) {
        block.ops.push_back({OpInvert});
        getNextToken();
      } else if (curTok == tok_grayscale) {
        block.ops.push_back({OpGrayscale});
        getNextToken();
      } else if (curTok == tok_rotate) {
        getNextToken();
        if (curTok != tok_number) {
          llvm::errs() << "Expected angle after 'rotate'\n";
          return false;
        }
        block.ops.push_back({OpRotate, lexer.numberVal});
        getNextToken();
      } else {
        break;
      }
    }

    operationsBlocks.push_back(block);
    return true;
  }

  OwningOpRef<ModuleOp> generateMLIR(MLIRContext &context) {
    // Create module
    auto loc = UnknownLoc::get(&context);
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    OpBuilder builder(module->getBodyRegion());

    // Create main function
    auto funcType = FunctionType::get(&context, {}, {});
    auto mainFunc = builder.create<func::FuncOp>(loc, "main", funcType);
    auto &entryBlock = *mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Map from image name to SSA value
    std::map<std::string, Value> imageMap;

    // Generate code for image declarations
    for (auto &decl : imageDecls) {
      auto imageType = ImageType::get(&context);
      auto loadOp = builder.create<LoadOp>(
          loc, imageType, builder.getStringAttr(decl.inputPath + ".bmp"));
      imageMap[decl.name] = loadOp.getResult();
    }

    // Generate code for operations blocks
    for (auto &block : operationsBlocks) {
      if (imageMap.find(block.imageName) == imageMap.end()) {
        llvm::errs() << "Unknown image: " << block.imageName << "\n";
        return nullptr;
      }

      Value currentImage = imageMap[block.imageName];
      auto imageType = ImageType::get(&context);

      // Apply operations sequentially
      for (auto &op : block.ops) {
        switch (op.type) {
        case OpInvert:
          currentImage = builder.create<InvertOp>(loc, imageType, currentImage).getResult();
          break;
        case OpGrayscale:
          currentImage = builder.create<GrayscaleOp>(loc, imageType, currentImage).getResult();
          break;
        case OpRotate:
          currentImage = builder.create<RotateOp>(
              loc, imageType, currentImage,
              builder.getF32FloatAttr(op.angle)).getResult();
          break;
        }
      }

      // Save the result
      for (auto &decl : imageDecls) {
        if (decl.name == block.imageName) {
          builder.create<SaveOp>(
              loc, currentImage,
              builder.getStringAttr(decl.outputPath + ".bmp"),
              builder.getStringAttr(decl.format));
          break;
        }
      }
    }

    // Add return
    builder.create<func::ReturnOp>(loc);

    return module;
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

mlir::OwningOpRef<mlir::ModuleOp> parsePixelScript(const std::string &scriptText,
                                                     mlir::MLIRContext &context) {
  Lexer lexer(scriptText);
  Parser parser(lexer);

  if (!parser.parseScript()) {
    llvm::errs() << "Failed to parse Pixel script\n";
    return nullptr;
  }

  return parser.generateMLIR(context);
}
