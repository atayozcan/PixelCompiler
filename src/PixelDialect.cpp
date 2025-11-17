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

//===----------------------------------------------------------------------===//
// Pixel Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "PixelOps.cpp.inc"
