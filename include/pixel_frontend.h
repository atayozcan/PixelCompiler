// pixel_frontend.h
#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

mlir::OwningOpRef<mlir::ModuleOp>
parsePixelScript(const std::string &scriptText, mlir::MLIRContext &context);
