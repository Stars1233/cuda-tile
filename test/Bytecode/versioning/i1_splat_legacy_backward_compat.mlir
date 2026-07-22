// Backward-compat regression: v13.3-era binaries wrote i1 splat
// DenseElementsAttr payloads as a single byte regardless of the tensor
// shape, with 0xff for true and 0x00 for false.
//
// RUN-1 checks the reader decodes the v13.3 bytes to the correct values.
// RUN-2 checks the writer targeting 13.3 reproduces the bytes exactly.

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.3/i1-splat-legacy-13.3.tileirbc | FileCheck %s

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.3/i1-splat-legacy-13.3.tileirbc -o %t.mlir
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.3 %t.mlir -o %t.bc
// RUN: cmp %t.bc %S/Inputs/13.3/i1-splat-legacy-13.3.tileirbc

// CHECK-LABEL: cuda_tile.module @kernels
// CHECK:   entry @i1_splat
// CHECK:     constant <i1: true> : tile<16xi1>
// CHECK:     constant <i1: false> : tile<16xi1>
