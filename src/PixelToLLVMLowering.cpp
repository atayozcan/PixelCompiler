//===- PixelToLLVMLowering.cpp - Lowering Pixel Dialect to LLVM ---------===//
//
// Implements lowering from Pixel dialect to LLVM dialect
//
//===----------------------------------------------------------------------===//
#include "PixelDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
using namespace mlir::pixel;
using namespace std;

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//
class PixelTypeConverter final : public LLVMTypeConverter {
public:
  explicit PixelTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    // Convert ImageType to opaque pointer (!llvm.ptr)
    addConversion([](const ImageType type) {
      return LLVM::LLVMPointerType::get(type.getContext());
    });
  }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//
namespace {
// Helper to create or get a runtime function declaration
LLVM::LLVMFuncOp getOrInsertFunction(PatternRewriter &rewriter, ModuleOp module,
                                     StringRef name,
                                     LLVM::LLVMFunctionType fnType) {
  if (const auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

/// Lower pixel.load to runtime call
struct LoadOpLowering final : ConvertOpToLLVMPattern<LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit LoadOpLowering(const LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter) {}

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();

    // Function signature: ptr @_pixel_load(ptr %filepath)
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    const auto fnType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto loadFn = getOrInsertFunction(rewriter, module, "_pixel_load", fnType);

    // Convert string to global constant
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

    // Get pointer to string
    auto globalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, ptrType, global.getSymName());

    // Call runtime function
    auto callOp =
        rewriter.create<LLVM::CallOp>(loc, loadFn, ValueRange{globalPtr});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

/// Lower pixel.save to runtime call
struct SaveOpLowering final : ConvertOpToLLVMPattern<SaveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit SaveOpLowering(const LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter) {}

  LogicalResult
  matchAndRewrite(SaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();

    // Function signature: void @_pixel_save(ptr %image, ptr %filepath)
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    const auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    const auto fnType =
        LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
    auto saveFn = getOrInsertFunction(rewriter, module, "_pixel_save", fnType);

    // Convert string to global constant
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

    auto globalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, ptrType, global.getSymName());

    // Call runtime function
    rewriter.create<LLVM::CallOp>(loc, saveFn,
                                  ValueRange{adaptor.getImage(), globalPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower pixel.invert to runtime call
struct InvertOpLowering final : ConvertOpToLLVMPattern<InvertOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit InvertOpLowering(const LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter) {}

  LogicalResult
  matchAndRewrite(InvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();

    // Function signature: ptr @_pixel_invert(ptr %image)
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    const auto fnType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto invertFn =
        getOrInsertFunction(rewriter, module, "_pixel_invert", fnType);

    auto callOp = rewriter.create<LLVM::CallOp>(loc, invertFn,
                                                ValueRange{adaptor.getInput()});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

/// Lower pixel.grayscale to runtime call
struct GrayscaleOpLowering final : ConvertOpToLLVMPattern<GrayscaleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit GrayscaleOpLowering(const LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter) {}

  LogicalResult
  matchAndRewrite(GrayscaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();

    // Function signature: ptr @_pixel_grayscale(ptr %image)
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    const auto fnType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto grayscaleFn =
        getOrInsertFunction(rewriter, module, "_pixel_grayscale", fnType);

    auto callOp = rewriter.create<LLVM::CallOp>(loc, grayscaleFn,
                                                ValueRange{adaptor.getInput()});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

/// Lower pixel.rotate to runtime call
struct RotateOpLowering final : ConvertOpToLLVMPattern<RotateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  explicit RotateOpLowering(const LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter) {}

  LogicalResult
  matchAndRewrite(RotateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto module = op->getParentOfType<ModuleOp>();
    const auto loc = op.getLoc();

    // Function signature: ptr @_pixel_rotate(ptr %image, float %angle)
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto floatType = Float32Type::get(rewriter.getContext());
    const auto fnType =
        LLVM::LLVMFunctionType::get(ptrType, {ptrType, floatType});
    auto rotateFn =
        getOrInsertFunction(rewriter, module, "_pixel_rotate", fnType);

    // Create constant for angle
    auto angleAttr = op.getAngle();
    auto angleConst =
        rewriter.create<LLVM::ConstantOp>(loc, floatType, angleAttr);

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, rotateFn, ValueRange{adaptor.getInput(), angleConst});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Lowering Pass
//===----------------------------------------------------------------------===//
namespace {
struct PixelToLLVMLoweringPass final
    : PassWrapper<PixelToLLVMLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PixelToLLVMLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    const auto module = getOperation();
    auto *context = &getContext();

    // Set up type converter
    PixelTypeConverter typeConverter(context);

    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<PixelDialect>();

    // Prepare function signature conversion
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    // Set up rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<LoadOpLowering, SaveOpLowering, InvertOpLowering,
                 GrayscaleOpLowering, RotateOpLowering>(typeConverter);

    // Add function signature conversion
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//
namespace mlir::pixel {
unique_ptr<Pass> createPixelToLLVMLoweringPass() {
  return make_unique<PixelToLLVMLoweringPass>();
}
} // namespace mlir::pixel
