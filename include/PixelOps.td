//===- PixelOps.td - Pixel dialect ops -----------*- tablegen -*-===//
//
// Defines the Pixel dialect operations for image processing
//
//===--------------------------------------------------------------===//

#ifndef PIXEL_OPS
#define PIXEL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Pixel dialect definition
def Pixel_Dialect : Dialect {
  let name = "pixel";
  let summary = "A high-level dialect for image processing operations";
  let description = [{
    The Pixel dialect provides high-level operations for image processing,
    including loading, saving, and transforming images.
  }];
  let cppNamespace = "::mlir::pixel";
}

// Base class for Pixel dialect ops
class Pixel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Pixel_Dialect, mnemonic, traits>;

// Image type definition
def Pixel_ImageType : Type<CPred<"llvm::isa<::mlir::pixel::ImageType>($_self)">,
                            "pixel image type">;

//===--------------------------------------------------------------===//
// Image Loading and Saving Operations
//===--------------------------------------------------------------===//

def Pixel_LoadOp : Pixel_Op<"load"> {
  let summary = "Load an image from file";
  let description = [{
    Loads an image from the specified file path.

    Example:
    ```mlir
    %img = pixel.load "input.bmp" : !pixel.image
    ```
  }];

  let arguments = (ins StrAttr:$filepath);
  let results = (outs Pixel_ImageType:$result);

  let assemblyFormat = "$filepath attr-dict `:` type($result)";
}

def Pixel_SaveOp : Pixel_Op<"save"> {
  let summary = "Save an image to file";
  let description = [{
    Saves an image to the specified file path with the given format.

    Example:
    ```mlir
    pixel.save %img, "output.bmp" {format = "bmp"} : !pixel.image
    ```
  }];

  let arguments = (ins
    Pixel_ImageType:$image,
    StrAttr:$filepath,
    StrAttr:$format
  );

  let assemblyFormat = "$image `,` $filepath attr-dict `:` type($image)";
}

//===--------------------------------------------------------------===//
// Image Processing Operations
//===--------------------------------------------------------------===//

def Pixel_InvertOp : Pixel_Op<"invert"> {
  let summary = "Invert image colors";
  let description = [{
    Inverts all pixel values in the image (255 - value for each channel).

    Example:
    ```mlir
    %inverted = pixel.invert %img : !pixel.image
    ```
  }];

  let arguments = (ins Pixel_ImageType:$input);
  let results = (outs Pixel_ImageType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Pixel_GrayscaleOp : Pixel_Op<"grayscale"> {
  let summary = "Convert image to grayscale";
  let description = [{
    Converts a color image to grayscale using standard luminosity method.
  }];

  let arguments = (ins Pixel_ImageType:$input);
  let results = (outs Pixel_ImageType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Pixel_RotateOp : Pixel_Op<"rotate"> {
  let summary = "Rotate image";
  let description = [{
    Rotates image by specified angle in degrees.
  }];

  let arguments = (ins
    Pixel_ImageType:$input,
    F32Attr:$angle
  );
  let results = (outs Pixel_ImageType:$result);

  let assemblyFormat = "$input `,` $angle attr-dict `:` type($input) `->` type($result)";
}

#endif // PIXEL_OPS
