# SeedVR Single GPU Optimization

This document describes the single GPU optimizations made to the SeedVR repository, which was originally designed for multi-GPU inference using distributed computing techniques.

## Overview

The original SeedVR repository uses several distributed computing strategies:
- **Distributed Data Parallel (DDP)** for multi-GPU training/inference
- **Sequence Parallel** for splitting long sequences across GPUs  
- **FSDP (Fully Sharded Data Parallel)** for model sharding
- **Complex distributed operations** for synchronization across GPUs

These optimizations remove the distributed overhead and simplify the codebase for efficient single GPU inference.

## Files Added/Modified

### New Files Created

1. **`common/distributed/single_gpu.py`**
   - Single GPU optimized functions that bypass all distributed operations
   - Replaces multi-GPU functions with single GPU equivalents
   - All distributed functions return single GPU defaults (rank=0, world_size=1)
   - No-op implementations for distributed synchronization functions

2. **`projects/inference_seedvr_3b_single_gpu.py`**
   - Single GPU optimized inference script for 3B model
   - Removes distributed partitioning and synchronization
   - Simplified batch processing without multi-GPU coordination
   - Uses single GPU configuration file

3. **`projects/inference_seedvr_7b_single_gpu.py`**
   - Single GPU optimized inference script for 7B model
   - Same optimizations as 3B version but for larger model
   - Includes memory management optimizations for larger model

4. **`configs_3b/main_single_gpu.yaml`**
   - Single GPU configuration for 3B model
   - Removes FSDP sharding strategy settings
   - Optimized for single GPU memory usage

5. **`configs_7b/main_single_gpu.yaml`**
   - Single GPU configuration for 7B model
   - Removes FSDP sharding strategy settings
   - Optimized for single GPU memory usage

## Key Optimizations

### 1. Distributed Operations Bypass
- All distributed functions in `single_gpu.py` return single GPU defaults
- No distributed initialization (`dist.init_process_group` skipped)
- No DDP wrapping of models
- No sequence parallel operations

### 2. Memory Management
- Maintains model offloading between CPU and GPU for memory efficiency
- Uses gradient checkpointing to reduce memory usage
- Optimized VAE memory limits for single GPU

### 3. Simplified Data Flow
- Removes distributed data partitioning
- No cross-GPU synchronization needed
- Simplified batch processing logic
- Direct tensor operations without distributed wrappers

### 4. Configuration Optimizations
- Removed FSDP sharding strategies from configs
- Maintained all model architecture settings
- Preserved diffusion and VAE configurations

## Usage

### For 3B Model
```bash
python projects/inference_seedvr_3b_single_gpu.py \
    --video_path ./test_videos \
    --output_dir ./results \
    --cfg_scale 6.5 \
    --sample_steps 50 \
    --seed 666 \
    --res_h 720 \
    --res_w 1280 \
    --batch_size 1
```

### For 7B Model
```bash
python projects/inference_seedvr_7b_single_gpu.py \
    --video_path ./test_videos \
    --output_dir ./results \
    --cfg_scale 6.5 \
    --sample_steps 50 \
    --seed 666 \
    --res_h 720 \
    --res_w 1280 \
    --batch_size 1
```

## Performance Benefits

### 1. Reduced Overhead
- No distributed communication overhead
- No synchronization barriers
- Faster startup time (no distributed initialization)

### 2. Simplified Debugging
- Easier to debug without distributed complexity
- Direct tensor operations
- Clearer error messages

### 3. Memory Efficiency
- No duplicate model copies across GPUs
- Optimized memory usage for single GPU
- Better memory management with offloading

### 4. Easier Deployment
- No need for multi-GPU setup
- Works on single GPU systems
- Simplified environment requirements

## Memory Requirements

### 3B Model
- **Minimum GPU Memory**: 16GB VRAM recommended
- **Optimal GPU Memory**: 24GB VRAM for comfortable operation
- Uses model offloading to manage memory usage

### 7B Model
- **Minimum GPU Memory**: 24GB VRAM recommended  
- **Optimal GPU Memory**: 40GB+ VRAM for comfortable operation
- Requires more aggressive memory management

## Technical Details

### Distributed Function Replacements
```python
# Original multi-GPU functions
get_world_size() -> returns actual world size
get_global_rank() -> returns actual rank
sync_data() -> performs cross-GPU synchronization

# Single GPU replacements
get_world_size() -> always returns 1
get_global_rank() -> always returns 0  
sync_data() -> returns data as-is (no-op)
```

### Model Loading Optimizations
- Models loaded directly to GPU without distributed wrappers
- No DDP conversion needed
- Simplified checkpoint loading

### Inference Pipeline
1. Load single GPU configuration
2. Initialize model without distributed setup
3. Process videos in simple batches
4. Generate without cross-GPU coordination
5. Save results directly

## Compatibility

### Maintained Compatibility
- All model architectures preserved
- Same inference quality as multi-GPU version
- Compatible with existing checkpoints
- Same output formats

### Removed Features
- Multi-GPU distributed inference
- Sequence parallel processing
- FSDP model sharding
- Cross-GPU synchronization

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size to 1
   - Lower resolution (res_h, res_w)
   - Ensure model offloading is working

2. **Slow Inference**
   - Check GPU utilization
   - Verify CUDA is properly installed
   - Monitor memory usage

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path includes project root
   - Verify checkpoint files exist

### Memory Optimization Tips
- Use smaller batch sizes for large models
- Monitor GPU memory usage during inference
- Consider using mixed precision (already enabled)
- Ensure adequate system RAM for model offloading

## Future Improvements

Potential areas for further optimization:
1. **Dynamic batch sizing** based on available memory
2. **Automatic resolution scaling** for memory constraints
3. **Progressive loading** for very large models
4. **Quantization support** for further memory reduction

## Conclusion

These single GPU optimizations make SeedVR accessible on single GPU systems while maintaining the same inference quality. The simplified codebase is easier to understand, debug, and deploy, making it more suitable for research and development environments with limited GPU resources.
