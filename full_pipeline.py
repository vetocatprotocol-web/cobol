#!/usr/bin/env python3
"""
Full L1–L8 Pipeline Orchestrator
Enables seamless compression/decompression through all 8 COBOL layers
Modes: LEGACY (L5–L7), BRIDGE (L1–L8), MAXIMAL (full L1–L8)
"""

import argparse
import sys
from pathlib import Path
from dual_mode_engine import DualModeEngine, CompressionMode


def print_stats(original: bytes, compressed: bytes, duration: float, mode: str):
    """Print compression statistics."""
    ratio = len(original) / len(compressed) if compressed else 0
    original_kb = len(original) / 1024
    compressed_kb = len(compressed) / 1024
    
    print(f"\n{'='*60}")
    print(f"Mode: {mode.upper()}")
    print(f"Original:   {len(original):>10} bytes ({original_kb:>8.2f} KB)")
    print(f"Compressed: {len(compressed):>10} bytes ({compressed_kb:>8.2f} KB)")
    print(f"Ratio:      {ratio:>10.2f}x")
    print(f"Duration:   {duration:>10.3f}s")
    print(f"Throughput: {len(original) / duration / 1024 / 1024:>10.2f} MB/s")
    print(f"{'='*60}\n")


def compress_file(input_file: Path, output_file: Path, mode: CompressionMode):
    """Compress a file using given mode."""
    import time
    
    # Read input
    print(f"Reading: {input_file}")
    with open(input_file, 'rb') as f:
        data = f.read()
    
    print(f"Input size: {len(data)} bytes")
    
    # Compress
    engine = DualModeEngine(mode)
    print(f"Compressing ({mode.value})...", end=' ')
    sys.stdout.flush()
    
    start = time.time()
    compressed = engine.compress(data)
    duration = time.time() - start
    
    # Write output
    with open(output_file, 'wb') as f:
        f.write(compressed)
    
    print(f"Done in {duration:.3f}s")
    print_stats(data, compressed, duration, mode.value)
    
    return len(data), len(compressed)


def decompress_file(input_file: Path, output_file: Path, mode: CompressionMode):
    """Decompress a file using given mode."""
    import time
    
    # Read input
    print(f"Reading: {input_file}")
    with open(input_file, 'rb') as f:
        data = f.read()
    
    print(f"Input size: {len(data)} bytes")
    
    # Decompress
    engine = DualModeEngine(mode)
    print(f"Decompressing ({mode.value})...", end=' ')
    sys.stdout.flush()
    
    start = time.time()
    decompressed = engine.decompress(data)
    duration = time.time() - start
    
    # Write output
    with open(output_file, 'wb') as f:
        f.write(decompressed)
    
    print(f"Done in {duration:.3f}s")
    print(f"\nDecompressed: {len(decompressed)} bytes → {output_file}")
    
    return len(data), len(decompressed)


def benchmark_modes(data: bytes, modes: list):
    """Benchmark different modes on same data."""
    import time
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: All Modes")
    print(f"Input: {len(data)} bytes")
    print(f"{'='*70}\n")
    
    results = []
    
    for mode in modes:
        print(f"Testing {mode.value.upper()}...", end=' ')
        sys.stdout.flush()
        
        engine = DualModeEngine(mode)
        
        start = time.time()
        compressed = engine.compress(data)
        compress_time = time.time() - start
        
        start = time.time()
        decompressed = engine.decompress(compressed)
        decompress_time = time.time() - start
        
        ratio = len(data) / len(compressed) if compressed else 0
        match = decompressed == data
        
        results.append({
            'mode': mode.value,
            'compressed': len(compressed),
            'ratio': ratio,
            'compress_time': compress_time,
            'decompress_time': decompress_time,
            'total_time': compress_time + decompress_time,
            'valid': match
        })
        
        status = "✓" if match else "✗"
        print(f"{status} {ratio:.2f}x in {compress_time + decompress_time:.3f}s")
    
    # Display summary
    print(f"\n{'='*70}")
    print(f"{'Mode':<15} {'Compressed':<15} {'Ratio':<15} {'Time':<15} {'Valid':<10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['mode']:<15} {r['compressed']:<15} {r['ratio']:<14.2f}x {r['total_time']:<14.3f}s {str(r['valid']):<10}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Full L1-L8 Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Compress a file (MAXIMAL mode):
    python3 full_pipeline.py compress input.txt -o output.bin --mode maximal
  
  Decompress a file:
    python3 full_pipeline.py decompress output.bin -o recovered.txt --mode maximal
  
  Benchmark all modes on a file:
    python3 full_pipeline.py benchmark input.txt
  
  Benchmark with synthetic data:
    python3 full_pipeline.py benchmark --synthetic 10000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Compress a file')
    compress_parser.add_argument('input', type=Path, help='Input file')
    compress_parser.add_argument('-o', '--output', type=Path, required=True, help='Output file')
    compress_parser.add_argument('--mode', choices=['legacy', 'bridge', 'maximal'], 
                                 default='maximal', help='Compression mode')
    
    # Decompress command
    decompress_parser = subparsers.add_parser('decompress', help='Decompress a file')
    decompress_parser.add_argument('input', type=Path, help='Input file')
    decompress_parser.add_argument('-o', '--output', type=Path, required=True, help='Output file')
    decompress_parser.add_argument('--mode', choices=['legacy', 'bridge', 'maximal'], 
                                   default='maximal', help='Compression mode')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark compression modes')
    benchmark_parser.add_argument('input', type=Path, nargs='?', help='Input file (optional)')
    benchmark_parser.add_argument('--synthetic', type=int, help='Use synthetic data of N bytes')
    benchmark_parser.add_argument('--modes', choices=['all', 'legacy', 'bridge', 'maximal'], 
                                  default='all', help='Which modes to benchmark')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'compress':
            if not args.input.exists():
                print(f"Error: Input file not found: {args.input}")
                return 1
            
            mode = CompressionMode[args.mode.upper()]
            compress_file(args.input, args.output, mode)
            
        elif args.command == 'decompress':
            if not args.input.exists():
                print(f"Error: Input file not found: {args.input}")
                return 1
            
            mode = CompressionMode[args.mode.upper()]
            decompress_file(args.input, args.output, mode)
            
        elif args.command == 'benchmark':
            if args.synthetic:
                # Use synthetic data
                print(f"Generating synthetic data: {args.synthetic} bytes")
                data = b"AAAA" * (args.synthetic // 4)  # Repetitive synthetic
            elif args.input:
                if not args.input.exists():
                    print(f"Error: Input file not found: {args.input}")
                    return 1
                print(f"Reading: {args.input}")
                with open(args.input, 'rb') as f:
                    data = f.read()
            else:
                print("Error: Either --synthetic or input file required")
                return 1
            
            # Determine which modes to test
            if args.modes == 'all':
                modes = [CompressionMode.LEGACY, CompressionMode.BRIDGE, CompressionMode.MAXIMAL]
            else:
                modes = [CompressionMode[args.modes.upper()]]
            
            benchmark_modes(data, modes)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
