#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="build"
BUILD_TYPE="Release"
CLEAN=false
VERBOSE=false
BUILD_DEMOS=false
BUILD_TESTS=true
BUILD_BENCHMARKS=false

usage() {
    cat <<EOF
Build helper for Tableau (header-only ODE library).

Options:
  -h, --help         Show this help
  -d, --debug        Configure Debug (default: Release)
  -c, --clean        Remove build directory before configuring
  -v, --verbose      Pass verbose build output
  --demos            Build interactive demos (requires OpenGL)
  --benchmarks       Build Google Benchmark suite (requires local google-benchmark)
  --no-tests         Skip building tests
  --build-dir DIR    Override build directory (default: build)

Examples:
  ./build.sh                          # Release build with tests
  ./build.sh --demos                  # Release build with demos
  ./build.sh --debug --demos          # Debug build with demos
  ./build.sh --benchmarks             # Release build with benchmarks
  ./build.sh --clean --debug          # Clean debug build
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -d|--debug) BUILD_TYPE="Debug"; shift ;;
        -c|--clean) CLEAN=true; shift ;;
        -v|--verbose) VERBOSE=true; shift ;;
        --demos) BUILD_DEMOS=true; shift ;;
        --benchmarks) BUILD_BENCHMARKS=true; shift ;;
        --no-tests) BUILD_TESTS=false; shift ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

BINARY_DIR="$SCRIPT_DIR/$BUILD_DIR"

echo "tableau build:"
echo "  type:        $BUILD_TYPE"
echo "  dir:         $BINARY_DIR"
echo "  clean:       $CLEAN"
echo "  verbose:     $VERBOSE"
echo "  demos:       $BUILD_DEMOS"
echo "  benchmarks:  $BUILD_BENCHMARKS"
echo "  tests:       $BUILD_TESTS"
echo

if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BINARY_DIR"
fi

CMAKE_ARGS=(
    -S "$SCRIPT_DIR"
    -B "$BINARY_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [ "$BUILD_DEMOS" = true ]; then
    CMAKE_ARGS+=(-DBUILD_DEMOS=ON)
fi

if [ "$BUILD_BENCHMARKS" = true ]; then
    CMAKE_ARGS+=(-DBUILD_BENCHMARKS=ON)
fi

if [ "$BUILD_TESTS" = false ]; then
    CMAKE_ARGS+=(-DBUILD_TESTING=OFF)
fi

cmake "${CMAKE_ARGS[@]}"

if [ "$VERBOSE" = true ]; then
    cmake --build "$BINARY_DIR" --config "$BUILD_TYPE" -- VERBOSE=1
else
    cmake --build "$BINARY_DIR" --config "$BUILD_TYPE"
fi

echo
echo "Done. Build artifacts in $BINARY_DIR."
if [ "$BUILD_DEMOS" = true ]; then
    echo "  Demos:"
    echo "    $BINARY_DIR/demos/three_body_demo"
    echo "    $BINARY_DIR/demos/black_hole_demo"
    echo "    $BINARY_DIR/demos/golf_demo"
fi
if [ "$BUILD_BENCHMARKS" = true ]; then
    echo "  Benchmarks:"
    echo "    $BINARY_DIR/tableau_benchmarks"
fi
