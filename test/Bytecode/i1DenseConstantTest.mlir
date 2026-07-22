// RUN: %round_trip_test %s %t

// i1 dense constants use a fixed wire encoding independent of the linked
// MLIR i1 raw-buffer layout: a splat is a single 0xff/0x00 byte and a
// non-splat is bit-packed. Exercise both across and beyond the
// 8-element-per-byte boundary.

cuda_tile.module @kernels {
  cuda_tile.entry @i1_dense_constants() {
    // Scalar splats.
    %0 = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    %1 = cuda_tile.constant <i1: 0> : !cuda_tile.tile<i1>
    // Non-splat, fits in one packed byte.
    %2 = cuda_tile.constant <i1: [true, false, true, false]> : !cuda_tile.tile<4xi1>
    %3 = cuda_tile.constant <i1: [false, false, true, true]> : !cuda_tile.tile<4xi1>
    // Exactly one packed byte (8 elements), non-splat and splat.
    %4 = cuda_tile.constant <i1: [true, false, false, false, false, false, false, true]> : !cuda_tile.tile<8xi1>
    %5 = cuda_tile.constant <i1: true> : !cuda_tile.tile<8xi1>
    // Multi-byte (>8 elements): splats and a non-splat with the endpoints set.
    %6 = cuda_tile.constant <i1: true> : !cuda_tile.tile<16xi1>
    %7 = cuda_tile.constant <i1: false> : !cuda_tile.tile<16xi1>
    %8 = cuda_tile.constant <i1: [true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]> : !cuda_tile.tile<16xi1>
    cuda_tile.return
  }
}
