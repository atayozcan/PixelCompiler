//===- PixelCompiler.cpp - Pixel Dialect and Lowering -------------------===//
//
// Combined implementation of Pixel dialect and lowering to LLVM
//
//===----------------------------------------------------------------------===//
#include "PixelDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
using namespace mlir::pixel;
using namespace std;

//===----------------------------------------------------------------------===//
// Pixel Dialect Implementation
//===----------------------------------------------------------------------===//
#include "PixelOpsDialect.cpp.inc"

void PixelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PixelOps.cpp.inc"
  >();
  addTypes<ImageType>();
}

void PixelDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (llvm::dyn_cast<ImageType>(type))
    printer << "image";
}

Type PixelDialect::parseType(DialectAsmParser &parser) const {
  if (parser.parseKeyword("image"))
    return {};
  return ImageType::get(getContext());
}

#define GET_OP_CLASSES
#include "PixelOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//
class PixelTypeConverter final : public LLVMTypeConverter {
public:
  explicit PixelTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    addConversion([](const ImageType type) {
      return LLVM::LLVMPointerType::get(type.getContext());
    });
  }
};

//===----------------------------------------------------------------------===//
// Lowering Helper
//===----------------------------------------------------------------------===//
LLVM::LLVMFuncOp getOrInsertFunction(PatternRewriter &rewriter, ModuleOp module,
                                     StringRef name, LLVM::LLVMFunctionType fnType) {
  if (const auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

//===----------------------------------------------------------------------===//
// Operation Lowering Patterns
//===----------------------------------------------------------------------===//
struct LoadOpLowering final : ConvertOpToLLVMPattern<LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit LoadOpLowering(const LLVMTypeConverter &typeConverter)
    : ConvertOpToLLVMPattern(typeConverter) {
  }

  LogicalResult matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    const auto fnType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto loadFn = getOrInsertFunction(rewriter, module, "_pixel_load", fnType);

    const auto filepath = op.getFilepath().str();
    auto globalName = "_str_" + to_string(reinterpret_cast<uintptr_t>(&op));
    LLVM::GlobalOp global;
    if (!((global = module.lookupSymbol<LLVM::GlobalOp>(globalName)))) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto arrayType = LLVM::LLVMArrayType::get(
        IntegerType::get(rewriter.getContext(), 8), filepath.size() + 1);
      global = rewriter.create<LLVM::GlobalOp>(
        loc, arrayType, true, LLVM::Linkage::Internal, globalName,
        rewriter.getStringAttr(filepath + '\0'));
    }
    auto globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, global.getSymName());
    auto callOp = rewriter.create<LLVM::CallOp>(loc, loadFn, ValueRange{globalPtr});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

struct SaveOpLowering final : ConvertOpToLLVMPattern<SaveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit SaveOpLowering(const LLVMTypeConverter &typeConverter)
    : ConvertOpToLLVMPattern(typeConverter) {
  }

  LogicalResult matchAndRewrite(SaveOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    const auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    const auto fnType = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
    auto saveFn = getOrInsertFunction(rewriter, module, "_pixel_save", fnType);

    const auto filepath = op.getFilepath().str();
    auto globalName = "_str_" + to_string(reinterpret_cast<uintptr_t>(&op));
    LLVM::GlobalOp global;
    if (!((global = module.lookupSymbol<LLVM::GlobalOp>(globalName)))) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto arrayType = LLVM::LLVMArrayType::get(
        IntegerType::get(rewriter.getContext(), 8), filepath.size() + 1);
      global = rewriter.create<LLVM::GlobalOp>(
        loc, arrayType, true, LLVM::Linkage::Internal, globalName,
        rewriter.getStringAttr(filepath + '\0'));
    }
    auto globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, global.getSymName());
    rewriter.create<LLVM::CallOp>(loc, saveFn, ValueRange{adaptor.getImage(), globalPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct InvertOpLowering final : ConvertOpToLLVMPattern<InvertOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit InvertOpLowering(const LLVMTypeConverter &typeConverter)
    : ConvertOpToLLVMPattern(typeConverter) {
  }

  LogicalResult matchAndRewrite(InvertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    const auto fnType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto invertFn = getOrInsertFunction(rewriter, module, "_pixel_invert", fnType);
    auto callOp = rewriter.create<LLVM::CallOp>(loc, invertFn, ValueRange{adaptor.getInput()});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

struct GrayscaleOpLowering final : ConvertOpToLLVMPattern<GrayscaleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit GrayscaleOpLowering(const LLVMTypeConverter &typeConverter)
    : ConvertOpToLLVMPattern(typeConverter) {
  }

  LogicalResult matchAndRewrite(GrayscaleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    const auto fnType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto grayscaleFn = getOrInsertFunction(rewriter, module, "_pixel_grayscale", fnType);
    auto callOp = rewriter.create<LLVM::CallOp>(loc, grayscaleFn, ValueRange{adaptor.getInput()});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

struct RotateOpLowering final : ConvertOpToLLVMPattern<RotateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit RotateOpLowering(const LLVMTypeConverter &typeConverter)
    : ConvertOpToLLVMPattern(typeConverter) {
  }

  LogicalResult matchAndRewrite(RotateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto floatType = Float32Type::get(rewriter.getContext());
    const auto fnType = LLVM::LLVMFunctionType::get(ptrType, {ptrType, floatType});
    auto rotateFn = getOrInsertFunction(rewriter, module, "_pixel_rotate", fnType);
    auto angleAttr = op.getAngle();
    auto angleConst = rewriter.create<LLVM::ConstantOp>(loc, floatType, angleAttr);
    auto callOp = rewriter.create<LLVM::CallOp>(loc, rotateFn, ValueRange{adaptor.getInput(), angleConst});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lowering Pass
//===----------------------------------------------------------------------===//
struct PixelToLLVMLoweringPass final
    : PassWrapper<PixelToLLVMLoweringPass, OperationPass<ModuleOp> > {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PixelToLLVMLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    const auto module = getOperation();
    auto *context = &getContext();

    PixelTypeConverter typeConverter(context);
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<PixelDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    RewritePatternSet patterns(context);
    patterns.add<LoadOpLowering, SaveOpLowering, InvertOpLowering,
      GrayscaleOpLowering, RotateOpLowering>(typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

namespace mlir::pixel {
  unique_ptr<Pass> createPixelToLLVMLoweringPass() {
    return make_unique<PixelToLLVMLoweringPass>();
  }
}
