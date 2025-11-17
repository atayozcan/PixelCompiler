//===- PixelPasses.h - Pixel Dialect Passes ------------------------------===//
// Declares passes for lowering Pixel dialect
//===----------------------------------------------------------------------===//
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::pixel {
    /// Create a pass to lower Pixel dialect to LLVM dialect
    std::unique_ptr<Pass> createPixelToLLVMLoweringPass();
} // namespace mlir::pixel
