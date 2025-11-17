//===- PixelDialect.h - Pixel Dialect Definition -------------------------===//
// Defines the Pixel dialect for image processing operations
//===----------------------------------------------------------------------===//
#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::pixel {
//===----------------------------------------------------------------------===//
// ImageType - Custom type for representing images
//===----------------------------------------------------------------------===//
class ImageType : public Type::TypeBase<ImageType, Type, TypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "pixel.image";

  static ImageType get(MLIRContext *context) { return Base::get(context); }
};
} // namespace mlir::pixel

// Include the generated dialect declarations
#include "PixelOpsDialect.h.inc"

// Include the generated operation class declarations
#define GET_OP_CLASSES
#include "PixelOps.h.inc"
