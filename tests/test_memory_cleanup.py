#!/usr/bin/env python3
"""
Test script to verify memory cleanup is working correctly.
This script simulates loading and unloading a model multiple times
to check if memory is properly released.
"""

import torch
import gc
import sys
import os

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from hyperparameter_search import cleanup_memory


def print_memory_stats(stage: str):
    """Print current GPU memory statistics."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"[{stage}] GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def simulate_trial():
    """Simulate loading a model and then cleaning it up."""
    print("\n--- Simulating model load ---")

    # Create a dummy large tensor to simulate model memory usage
    dummy_model = torch.randn(1000, 1000, 100, device='cuda')  # ~400 MB

    print_memory_stats("After model load")

    # Simulate some operations
    _ = dummy_model * 2

    # Delete the model
    del dummy_model

    print_memory_stats("After del")

    # Cleanup memory
    cleanup_memory()

    print_memory_stats("After cleanup_memory()")


def main():
    """Run memory cleanup test."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires a GPU.")
        return

    print("="*80)
    print("Memory Cleanup Test")
    print("="*80)
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print("="*80)

    # Check if expandable_segments is set
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
    print(f"PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")
    print("="*80)

    print_memory_stats("Initial")

    # Run multiple simulations
    num_trials = 5
    for i in range(num_trials):
        print(f"\n{'='*80}")
        print(f"Trial {i+1}/{num_trials}")
        print('='*80)
        simulate_trial()

    print(f"\n{'='*80}")
    print("Final Memory State")
    print('='*80)
    print_memory_stats("Final")

    # Check if memory has accumulated
    final_allocated = torch.cuda.memory_allocated() / 1024**3
    if final_allocated < 0.1:  # Less than 100 MB
        print(f"\n✅ SUCCESS: Memory properly cleaned up ({final_allocated:.3f} GB remaining)")
    else:
        print(f"\n⚠️  WARNING: Some memory still allocated ({final_allocated:.3f} GB)")
        print("   This might indicate a memory leak, but could also be normal CUDA overhead.")


if __name__ == "__main__":
    # Set the environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
