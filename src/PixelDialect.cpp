//===- PixelDialect.cpp - Pixel Dialect Implementation ------------------===//
#include "PixelDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
using namespace mlir;
using namespace mlir::pixel;

//===----------------------------------------------------------------------===//
// Pixel Dialect
//===----------------------------------------------------------------------===//
#include "PixelOpsDialect.cpp.inc"

void PixelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PixelOps.cpp.inc"

      >();
  addTypes<ImageType>();
}

// Type printing
void PixelDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (llvm::dyn_cast<ImageType>(type))
    printer << "image";
}

// Type parsing
Type PixelDialect::parseType(DialectAsmParser &parser) const {
  if (parser.parseKeyword("image"))
    return Type();
  return ImageType::get(getContext());
}

//===----------------------------------------------------------------------===//
// Pixel Operations
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "PixelOps.cpp.inc"
