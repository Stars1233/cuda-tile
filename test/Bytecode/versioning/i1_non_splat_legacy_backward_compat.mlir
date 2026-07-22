// Backward-compat regression: bit-packed non-splat tile<NxI1> payloads must
// read and write byte-identically at v13.3. Non-splats are bit-packed
// (ceil(N/8) bytes, element i in bit (i % 8) of byte (i / 8), least-
// significant bit first).
//
// The committed i1-non-splat-legacy-13.3.tileirbc patterns catch bit-order
// and byte-order regressions:
//   * i1_alt_4        : [T,F,T,F] -- alternating, 1 byte (0x05).
//   * i1_ff_tt        : [F,F,T,T] -- asymmetric (0x0c vs 0x03).
//   * i1_endpoints_16 : [T, F*14, T] -- 16 elements, two bytes (0x01 0x80),
//                                       so bit 0 lands in byte 0 and bit 15
//                                       in byte 1.
//
// RUN-1 checks the reader decodes the v13.3 bytes to the correct values.
// RUN-2 checks the writer targeting 13.3 reproduces the bytes exactly.

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.3/i1-non-splat-legacy-13.3.tileirbc | FileCheck %s

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.3/i1-non-splat-legacy-13.3.tileirbc -o %t.mlir
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.3 %t.mlir -o %t.bc
// RUN: cmp %t.bc %S/Inputs/13.3/i1-non-splat-legacy-13.3.tileirbc

// CHECK-LABEL: cuda_tile.module @kernels
// CHECK:   entry @i1_alt_4
// CHECK:     constant <i1: [true, false, true, false]> : tile<4xi1>
// CHECK:   entry @i1_ff_tt
// CHECK:     constant <i1: [false, false, true, true]> : tile<4xi1>
// CHECK:   entry @i1_endpoints_16
// CHECK:     constant <i1: [true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]> : tile<16xi1>
