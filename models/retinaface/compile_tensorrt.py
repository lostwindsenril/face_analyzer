#!/usr/bin/env python3
"""
RetinaFace TensorRT FP16 compilation script.
Uses native TensorRT API to convert PyTorch models to TensorRT engines.
Supports dynamic input shapes and FP16 optimization.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from loguru import logger

# TensorRT log level
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class RetinaFaceTensorRTCompiler:
    """RetinaFace TensorRT compiler."""
    
    def __init__(self, model_path: str, output_dir: str = None):
        """
        Initialize the compiler.

        Args:
            model_path: PyTorch model path
            output_dir: TensorRT engine output directory
        """
        self.model_path = model_path
        if output_dir:
            self.output_dir = output_dir
        else:
            model_dir = os.path.dirname(model_path)
            self.output_dir = model_dir if model_dir else "."
        self.engine_path = os.path.join(self.output_dir, "retinaface_fp16.trt")
        self.onnx_path = os.path.join(self.output_dir, "retinaface_temp.onnx")
        
        # Dynamic input config - supports 1000x1000 resolution
        self.min_batch_size = 1
        self.opt_batch_size = 1
        self.max_batch_size = 4

        self.min_height = 320
        self.opt_height = 640
        self.max_height = 1000  # Updated to support 1000x1000 resolution

        self.min_width = 320
        self.opt_width = 640
        self.max_width = 1000   # Updated to support 1000x1000 resolution
        
        # Load config
        self.cfg = self._load_config()
        
        logger.info(f"TensorRT编译器初始化完成")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"引擎路径: {self.engine_path}")
    
    def _load_config(self):
        """Load RetinaFace config."""
        config_path = self.model_path.replace('pytorch_model.pt', 'configuration.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['models']
    
    def export_to_onnx(self):
        """Export PyTorch model to ONNX."""
        logger.info("开始导出ONNX模型...")
        
        try:
            # Import RetinaFace dynamically to avoid circular imports
            from models.retinaface import RetinaFace

            # Load PyTorch model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = RetinaFace(cfg=self.cfg)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove module. prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            net.load_state_dict(new_state_dict, strict=False)
            net.to(device)
            net.eval()
            
            # Create dynamic input
            dummy_input = torch.randn(1, 3, self.opt_height, self.opt_width).to(device)
            
            # Export ONNX
            torch.onnx.export(
                net,
                dummy_input,
                self.onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['loc', 'conf', 'landms'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'loc': {0: 'batch_size'},
                    'conf': {0: 'batch_size'},
                    'landms': {0: 'batch_size'}
                }
            )
            
            logger.info(f"ONNX模型导出成功: {self.onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def build_tensorrt_engine(self):
        """Build TensorRT engine."""
        logger.info("开始构建TensorRT引擎...")
        
        try:
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(self.onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("ONNX解析失败")
                    for error in range(parser.num_errors):
                        logger.error(f"解析错误: {parser.get_error(error)}")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
            
            # Force FP16 precision
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("强制启用FP16精度优化")

            # Check platform FP16 support
            if builder.platform_has_fast_fp16:
                logger.info("平台支持快速FP16计算")
            else:
                logger.warning("平台FP16支持有限，但仍使用FP16精度")
            
            # Set dynamic input shapes
            profile = builder.create_optimization_profile()
            
            # Input tensor shape config
            profile.set_shape(
                "input",
                (self.min_batch_size, 3, self.min_height, self.min_width),  # min
                (self.opt_batch_size, 3, self.opt_height, self.opt_width),  # opt
                (self.max_batch_size, 3, self.max_height, self.max_width)   # max
            )
            
            config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("正在构建TensorRT引擎，这可能需要几分钟...")
            start_time = time.time()

            serialized_engine = builder.build_serialized_network(network, config)
            
            build_time = time.time() - start_time
            logger.info(f"TensorRT引擎构建完成，耗时: {build_time:.2f}秒")

            if serialized_engine is None:
                logger.error("TensorRT引擎构建失败")
                return False

            # Save engine
            with open(self.engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            logger.info(f"TensorRT引擎保存成功: {self.engine_path}")
            
            # Clean up temporary ONNX file
            if os.path.exists(self.onnx_path):
                os.remove(self.onnx_path)
                logger.info("临时ONNX文件已清理")
            
            return True
            
        except Exception as e:
            logger.error(f"TensorRT引擎构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_engine(self):
        """Validate TensorRT engine."""
        logger.info("开始验证TensorRT引擎...")
        
        try:
            # Load engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if engine is None:
                logger.error("引擎加载失败")
                return False
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Validate dynamic input, including 1000x1000
            test_shapes = [
                (1, 3, 320, 320),
                (1, 3, 640, 640),
                (2, 3, 800, 800),
                (1, 3, 1000, 1000)  # Validate 1000x1000 resolution
            ]
            
            for shape in test_shapes:
                try:
                    # Set input shape
                    context.set_binding_shape(0, shape)
                    
                    # Verify shape validity
                    if not context.all_binding_shapes_specified:
                        logger.warning(f"形状 {shape} 无效")
                        continue
                    
                    logger.info(f"形状 {shape} 验证通过")
                    
                except Exception as e:
                    logger.warning(f"形状 {shape} 验证失败: {e}")
            
            logger.info("TensorRT引擎验证完成")
            return True
            
        except Exception as e:
            logger.error(f"引擎验证失败: {e}")
            return False
    
    def benchmark_performance(self, num_iterations=100):
        """Performance benchmark."""
        logger.info("开始性能基准测试...")
        
        try:
            # Load engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # Test different input sizes, including 1000x1000
            test_cases = [
                (1, 3, 320, 320),
                (1, 3, 640, 640),
                (1, 3, 800, 800),
                (1, 3, 1000, 1000)  # Performance test at 1000x1000
            ]
            
            results = {}
            
            for shape in test_cases:
                batch_size, channels, height, width = shape
                
                # Set input shape
                context.set_input_shape("input", shape)
                
                # Allocate GPU memory
                input_size = int(batch_size * channels * height * width * 4)  # float32
                d_input = cuda.mem_alloc(input_size)
                
                # Get output shapes and allocate memory
                output_shapes = []
                d_outputs = []
                
                for i in range(1, engine.num_io_tensors):
                    tensor_name = engine.get_tensor_name(i)
                    output_shape = context.get_tensor_shape(tensor_name)
                    output_size = int(np.prod(output_shape) * 4)  # float32
                    d_output = cuda.mem_alloc(output_size)
                    d_outputs.append(d_output)
                    output_shapes.append(output_shape)
                
                # Create input data
                input_data = np.random.randn(*shape).astype(np.float32)
                
                # Set tensor addresses
                context.set_tensor_address("input", int(d_input))
                for i, d_output in enumerate(d_outputs):
                    tensor_name = engine.get_tensor_name(i + 1)
                    context.set_tensor_address(tensor_name, int(d_output))

                # Warm-up
                for _ in range(10):
                    cuda.memcpy_htod(d_input, input_data)
                    context.execute_async_v3(0)  # 使用stream 0
                    cuda.Context.synchronize()

                # Performance test
                start_time = time.time()

                for _ in range(num_iterations):
                    cuda.memcpy_htod(d_input, input_data)
                    context.execute_async_v3(0)  # 使用stream 0
                    cuda.Context.synchronize()
                
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_iterations * 1000  # ms
                fps = 1000 / avg_time
                
                results[f"{height}x{width}"] = {
                    "avg_time_ms": avg_time,
                    "fps": fps,
                    "shape": shape
                }
                
                logger.info(f"输入尺寸 {height}x{width}: {avg_time:.2f}ms, {fps:.1f} FPS")
                
                # Free memory
                d_input.free()
                for d_output in d_outputs:
                    d_output.free()
            
            # Save performance report
            report_path = os.path.join(self.output_dir, "tensorrt_performance_report.json")
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"性能报告已保存: {report_path}")
            return results
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compile(self):
        """Full compilation flow."""
        logger.info("开始RetinaFace TensorRT编译流程...")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Export ONNX
        if not self.export_to_onnx():
            return False
        
        # 2. Build TensorRT engine
        if not self.build_tensorrt_engine():
            return False
        
        # 3. Validate engine
        if not self.validate_engine():
            return False
        
        # 4. Performance test
        performance_results = self.benchmark_performance()
        
        if performance_results:
            logger.info("TensorRT编译完成！")
            logger.info("性能摘要:")
            for size, metrics in performance_results.items():
                logger.info(f"  {size}: {metrics['avg_time_ms']:.2f}ms, {metrics['fps']:.1f} FPS")
            return True
        else:
            logger.warning("编译完成，但性能测试失败")
            return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="RetinaFace TensorRT编译器")
    parser.add_argument("--model_path", type=str, 
                       default="models/retinaface/pytorch_model.pt",
                       help="PyTorch模型路径")
    parser.add_argument("--output_dir", type=str,
                       default=None,
                       help="输出目录")
    parser.add_argument("--benchmark", action="store_true",
                       help="只运行性能测试")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    # Check model file
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return False
    
    # Create compiler
    compiler = RetinaFaceTensorRTCompiler(args.model_path, args.output_dir)
    
    if args.benchmark:
        # Run performance test only
        if os.path.exists(compiler.engine_path):
            compiler.benchmark_performance()
        else:
            logger.error(f"TensorRT引擎不存在: {compiler.engine_path}")
    else:
        # Full compile flow
        success = compiler.compile()
        if success:
            logger.info("编译成功！")
            return True
        else:
            logger.error("编译失败！")
            return False


class TensorRTInferenceEngine:
    """Optimized TensorRT inference engine."""

    def __init__(self, engine_path: str):
        """
        Initialize the inference engine.

        Args:
            engine_path: TensorRT engine file path
        """
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.runtime = None
        self.stream = None

        # Optimized memory management
        self.buffer_cache = {}  # Cache buffers of different sizes
        self.current_input_shape = None
        self.d_input = None
        self.d_outputs = []
        self.h_outputs = []

        self._load_engine()

    def _load_engine(self):
        """Load TensorRT engine."""
        try:
            # Load engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()

            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

            # Create CUDA stream
            self.stream = cuda.Stream()

            logger.info(f"TensorRT引擎加载成功: {self.engine_path}")

        except Exception as e:
            logger.error(f"TensorRT引擎加载失败: {e}")
            raise

    def _allocate_buffers(self, input_shape):
        """Optimized buffer allocation with reuse."""
        shape_key = tuple(input_shape)

        # Check if existing buffers can be reused
        if shape_key == self.current_input_shape and self.d_input is not None:
            return  # Reuse existing buffers

        # Free old buffers
        self._free_buffers()

        # Set input shape
        self.context.set_input_shape("input", input_shape)

        # Allocate input buffer
        input_size = int(np.prod(input_shape) * 4)  # float32
        self.d_input = cuda.mem_alloc(input_size)
        self.context.set_tensor_address("input", int(self.d_input))

        # Allocate output buffers
        self.d_outputs = []
        self.h_outputs = []

        for i in range(1, self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            output_shape = self.context.get_tensor_shape(tensor_name)
            output_size = int(np.prod(output_shape) * 4)  # float32

            # GPU memory
            d_output = cuda.mem_alloc(output_size)
            self.d_outputs.append(d_output)
            self.context.set_tensor_address(tensor_name, int(d_output))

            # CPU memory (based on engine precision)
            # Check tensor dtype
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            if tensor_dtype == trt.DataType.HALF:
                h_output = np.empty(output_shape, dtype=np.float16)
            else:
                h_output = np.empty(output_shape, dtype=np.float32)
            self.h_outputs.append(h_output)

        # Record current shape
        self.current_input_shape = shape_key

    def _free_buffers(self):
        """Free buffers."""
        if self.d_input:
            self.d_input.free()
            self.d_input = None

        for d_output in self.d_outputs:
            if d_output:
                d_output.free()

        self.d_outputs = []
        self.h_outputs = []

    def infer(self, input_data):
        """
        Optimized inference execution.

        Args:
            input_data: Input numpy array with shape (batch, channels, height, width)

        Returns:
            tuple: (loc, conf, landms) inference outputs
        """
        input_shape = input_data.shape

        # Ensure input data is contiguous
        if not input_data.flags['C_CONTIGUOUS']:
            input_data = np.ascontiguousarray(input_data)

        # Allocate buffers (with reuse)
        self._allocate_buffers(input_shape)

        # Copy input to GPU
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)

        # Execute inference
        self.context.execute_async_v3(self.stream.handle)

        # Copy outputs to CPU
        for d_output, h_output in zip(self.d_outputs, self.h_outputs):
            cuda.memcpy_dtoh_async(h_output, d_output, self.stream)

        # Synchronize stream
        self.stream.synchronize()

        # Return results
        return tuple(self.h_outputs)

    def __del__(self):
        """Destructor to ensure memory is freed."""
        try:
            self._free_buffers()
        except:
            pass  # Ignore destructor errors

    def __del__(self):
        """Destructor to clean up resources."""
        self._free_buffers()

        if self.stream:
            self.stream = None


if __name__ == "__main__":
    main()
