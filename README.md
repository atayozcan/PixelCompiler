# Pixel Compiler

A simplified compiler for the Pixel image processing DSL, built with MLIR.

## What It Does

Compiles `.px` scripts that process BMP images using operations like:
- `invert` - Inverts colors
- `grayscale` - Converts to grayscale
- `rotate` - Rotates image by specified degrees

## Building

```bash
./build.sh
```

Or manually:
```bash
mkdir build && cd build
cmake ..
make -j4
```

## Usage

```bash
./compile.sh script.px
```

That's it! The compiler automatically:
1. Parses your `.px` file
2. Generates high-level MLIR (Pixel dialect)
3. Lowers to LLVM dialect MLIR
4. Generates LLVM IR
5. Compiles to executable

### Output Files

- `output_high_level.mlir` - High-level Pixel dialect
- `output_low_level.mlir` - Low-level LLVM dialect
- `output.ll` - LLVM IR
- `pixel_program` - Ready-to-run executable

Run the generated program:
```bash
./pixel_program
```

## Example Script

```
# script.px
image img1 "input" -> "output" as bmp

operations img1:
  invert
  grayscale
  rotate 90
```
