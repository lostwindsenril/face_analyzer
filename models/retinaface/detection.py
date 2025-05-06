# The implementation is based on resnet, available at https://github.com/biubug6/Pytorch_Retinaface
import numpy as np
import torch
import os
import json

from .utils import PriorBox, decode, decode_landm, py_cpu_nms

# TensorRT-related imports
import tensorrt as trt
import pycuda.driver as cuda

from loguru import logger

# TensorRT log level
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTInferenceEngine:
    """Built-in TensorRT inference engine (GPU-optimized)."""

    def __init__(self, engine_path: str, device='cuda'):
        """
        Initialize the inference engine.

        Args:
            engine_path: TensorRT engine file path
            device: Device type
        """
        self.engine_path = engine_path
        self.device = device
        self.engine = None
        self.context = None
        self.runtime = None
        self.stream = None

        # Preallocated buffer config
        self.max_input_shape = (1, 3, 1000, 1000)  # Max 1000x1000 resolution
        self.d_input = None
        self.d_outputs = []
        self.h_outputs = []

        self._load_engine()
        self._preallocate_buffers()

    def _load_engine(self):
        """Load TensorRT engine."""
        try:
            # Ensure CUDA context
            import pycuda.autoinit  # noqa: F401 Ensure CUDA context init

            # Load engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()

            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

            # Create CUDA stream
            self.stream = cuda.Stream()

        except Exception as e:
            logger.error(f"TensorRT引擎加载失败: {e}")
            raise

    def _preallocate_buffers(self):
        try:
            # Set max input shape
            self.context.set_input_shape("input", self.max_input_shape)

            # Check input tensor precision
            input_dtype = self.engine.get_tensor_dtype("input")
            if input_dtype == trt.DataType.HALF:
                input_element_size = 2  # FP16 = 2 bytes
                logger.info("TensorRT引擎使用FP16输入精度")
            else:
                input_element_size = 4  # FP32 = 4 bytes
                logger.warning("TensorRT引擎使用FP32输入精度，建议使用FP16引擎")

            # Preallocate input buffer based on precision
            max_input_size = int(np.prod(self.max_input_shape) * input_element_size)
            self.d_input = cuda.mem_alloc(max_input_size)
            self.context.set_tensor_address("input", int(self.d_input))

            # Preallocate output buffers
            self.d_outputs = []
            self.h_outputs = []

            for i in range(1, self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                output_shape = self.context.get_tensor_shape(tensor_name)

                # Allocate memory based on output precision
                tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
                if tensor_dtype == trt.DataType.HALF:
                    output_size = int(np.prod(output_shape) * 2)  # FP16 = 2 bytes
                    h_output = np.empty(output_shape, dtype=np.float16)
                    logger.info(f"输出张量 {tensor_name} 使用FP16精度")
                else:
                    output_size = int(np.prod(output_shape) * 4)  # FP32 = 4 bytes
                    h_output = np.empty(output_shape, dtype=np.float32)
                    logger.warning(f"输出张量 {tensor_name} 使用FP32精度")

                # GPU memory allocation
                d_output = cuda.mem_alloc(output_size)
                self.d_outputs.append(d_output)
                self.context.set_tensor_address(tensor_name, int(d_output))
                self.h_outputs.append(h_output)

            logger.info(f"预分配缓冲区完成，最大支持尺寸: {self.max_input_shape}")
            logger.info(f"输入精度: {'FP16' if input_dtype == trt.DataType.HALF else 'FP32'}")

        except Exception as e:
            logger.error(f"预分配缓冲区失败: {e}")
            raise

    def _validate_input_shape(self, input_shape):
        """Validate input shape within preallocated range."""
        max_batch, max_channels, max_height, max_width = self.max_input_shape
        batch, channels, height, width = input_shape

        if (batch > max_batch or channels > max_channels or
            height > max_height or width > max_width):
            raise ValueError(
                f"输入尺寸 {input_shape} 超出预分配的最大尺寸 {self.max_input_shape}"
            )

        return True

    def _free_buffers(self):
        """Free preallocated buffers."""
        if self.d_input:
            self.d_input.free()
            self.d_input = None

        for d_output in self.d_outputs:
            if d_output:
                d_output.free()

        self.d_outputs = []
        self.h_outputs = []

    def infer_from_gpu_tensor(self, gpu_tensor):
        """
        Run FP16 GPU inference using preallocated buffers.

        Args:
            gpu_tensor: GPU torch tensor with shape (batch, channels, height, width)

        Returns:
            tuple: (loc, conf, landms) GPU torch.float16 outputs
        """
        input_shape = gpu_tensor.shape

        # Set current input shape (dynamic output shapes)
        self.context.set_input_shape("input", input_shape)

        # Ensure all binding shapes are specified (TensorRT dynamic shape requirement)
        if not self.context.all_binding_shapes_specified:
            logger.error("并非所有绑定形状都已指定")
            raise RuntimeError("TensorRT上下文绑定形状未完全指定")

        # Ensure tensor is contiguous and FP16 (TensorRT FP16 engine requirement)
        if not gpu_tensor.is_contiguous():
            gpu_tensor = gpu_tensor.contiguous()

        # Convert to FP16 (TensorRT FP16 engine requirement)
        if gpu_tensor.dtype != torch.float16:
            gpu_tensor = gpu_tensor.to(torch.float16)

        # Optimized GPU memory copy: balance performance and compatibility
        # Remove unnecessary conversions to reduce transfer overhead

        # Ensure tensor is contiguous with correct precision
        if not gpu_tensor.is_contiguous():
            gpu_tensor = gpu_tensor.contiguous()

        # Smart precision conversion to avoid FP16->FP32->FP16
        if gpu_tensor.dtype != torch.float16:
            gpu_tensor = gpu_tensor.to(torch.float16)

        # Optimized memory transfer (reduce steps, keep compatibility)
        gpu_tensor_cpu = gpu_tensor.detach().cpu().numpy()
        cuda.memcpy_htod_async(self.d_input, gpu_tensor_cpu, self.stream)

        # Validate execution context status
        logger.debug(f"执行推理前验证 - 输入形状: {input_shape}")
        logger.debug(f"绑定形状已指定: {self.context.all_binding_shapes_specified}")

        # Execute inference
        self.context.execute_async_v3(self.stream.handle)

        # Copy outputs to GPU tensors (FP16 output handling)
        gpu_outputs = []
        for i, d_output in enumerate(self.d_outputs):
            # Get actual output shape (dynamic shape support)
            tensor_name = self.engine.get_tensor_name(i + 1)

            # Ensure output shape is computed correctly
            try:
                actual_output_shape = self.context.get_tensor_shape(tensor_name)
                logger.debug(f"输出张量 {tensor_name} 形状: {actual_output_shape}")
            except Exception as e:
                logger.error(f"获取输出张量 {tensor_name} 形状失败: {e}")
                raise RuntimeError(f"无法获取输出张量形状: {tensor_name}")

            # Validate output shape
            if any(dim <= 0 for dim in actual_output_shape):
                logger.error(f"输出张量 {tensor_name} 形状无效: {actual_output_shape}")
                raise RuntimeError(f"输出张量形状无效: {tensor_name}")

            # Validate TensorRT output precision
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)

            # Convert TensorRT dims to tuple
            shape_tuple = tuple(actual_output_shape)

            if tensor_dtype == trt.DataType.HALF:
                # FP16 output - create GPU tensor directly
                gpu_output = torch.empty(shape_tuple, dtype=torch.float16, device=self.device)
                logger.debug(f"输出张量 {tensor_name} 使用FP16精度")
            else:
                # If engine isn't FP16, create FP32 then convert to FP16
                gpu_output = torch.empty(shape_tuple, dtype=torch.float32, device=self.device)
                logger.warning(f"输出张量 {tensor_name} 不是FP16精度，将进行转换")

            # Copy from GPU buffer (avoid extra CPU transfer)
            # Create temp CPU array then copy to GPU tensor
            temp_cpu_array = np.empty(shape_tuple, dtype=np.float32 if tensor_dtype != trt.DataType.HALF else np.float16)
            cuda.memcpy_dtoh_async(temp_cpu_array, d_output, self.stream)

            gpu_outputs.append((temp_cpu_array, tensor_dtype))

        # Synchronize stream to finish transfers
        self.stream.synchronize()

        # Convert CPU arrays to GPU tensors
        final_outputs = []
        for temp_cpu_array, tensor_dtype in gpu_outputs:
            # Convert CPU array to GPU tensor
            gpu_output = torch.from_numpy(temp_cpu_array).to(device=self.device)

            # Ensure FP16 output
            if gpu_output.dtype != torch.float16:
                gpu_output = gpu_output.to(torch.float16)

            final_outputs.append(gpu_output)

        # Return GPU tensors, keep full GPU data flow
        return tuple(final_outputs)

    def __del__(self):
        """Destructor to ensure memory is freed."""
        try:
            self._free_buffers()
        except:
            pass


class RetinaFaceDetection(torch.nn.Module):

    def __init__(self, model_path, device='cuda'):
        # Initialize torch.nn.Module
        torch.nn.Module.__init__(self)
        # Initialize model attributes
        self.model_dir = model_path
        self.model_path = model_path
        self.device = device

        # Load configuration
        config_path = model_path.replace('pytorch_model.pt', 'configuration.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        self.cfg = config_data['models']

        self.trt_engine = None
        self._init_tensorrt()

        # GPU-accelerated preprocessing config
        self.use_fp16 = True
        self.torch_dtype = torch.float16

        # Preprocessing parameters on GPU
        self.mean_tensor = torch.tensor([104, 117, 123], dtype=self.torch_dtype, device=self.device).view(1, 3, 1, 1)

        # GPU memory buffer management
        self.gpu_buffers = {}
        self.max_buffer_size = 2048  # Support up to 2048x2048 images

    def _init_tensorrt(self):
        """Initialize TensorRT inference engine."""
        engine_path = self.model_path.replace('pytorch_model.pt', 'retinaface_fp16.trt')

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT引擎文件不存在: {engine_path}")

        try:
            logger.info(f"加载TensorRT引擎: {engine_path}")
            self.trt_engine = TensorRTInferenceEngine(engine_path)
            logger.info("TensorRT引擎初始化成功")
        except Exception as e:
            logger.error(f"TensorRT引擎初始化失败: {e}")
            raise RuntimeError(f"TensorRT引擎初始化失败: {e}")

    def forward(self, input):
        """GPU-accelerated forward inference."""

        img_input = input['img']
        # Convert numpy array to GPU tensor: [H, W, C] -> [1, C, H, W]
        img_tensor = torch.from_numpy(img_input).to(device=self.device, dtype=self.torch_dtype)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [H,W,C] -> [1,C,H,W]

        # Get original size
        _, _, im_height, im_width = img_tensor.shape

        max_dim = max(im_height, im_width)
        ss = torch.tensor(1000.0, dtype=self.torch_dtype, device=self.device) / torch.tensor(max_dim, dtype=self.torch_dtype, device=self.device)

        # Compute new size
        new_height = int(im_height * ss.item())
        new_width = int(im_width * ss.item())

        # GPU-accelerated resize with torch.nn.functional.interpolate
        # Use area mode to better match OpenCV default behavior
        img_tensor = torch.nn.functional.interpolate(
            img_tensor,
            size=(new_height, new_width),
            mode='area'
        )

        # Update size
        _, _, im_height, im_width = img_tensor.shape

        # GPU-accelerated preprocessing
        # Subtract mean (broadcast)
        img_processed = img_tensor - self.mean_tensor

        # Run TensorRT inference directly on GPU tensor
        loc, conf, landms = self.trt_engine.infer_from_gpu_tensor(img_processed)

        # Post-processing
        scale_factor = ss.item() if ss is not None else 1.0
        return self._postprocess(loc, conf, landms, im_height, im_width, scale_factor)

    def _postprocess(self, loc, conf, landms, im_height, im_width, ss):
        """Optimized postprocess (GPU tensors, CPU numpy output)."""
        # Use thresholds in original precision
        confidence_threshold = 0.9
        nms_threshold = 0.4
        top_k = 5000
        keep_top_k = 750

        # Create scale tensors (FP16 on GPU)
        scale = torch.tensor([im_width, im_height, im_width, im_height],
                           dtype=torch.float16, device=self.device)
        scale1 = torch.tensor([
            im_width, im_height, im_width, im_height, im_width, im_height,
            im_width, im_height, im_width, im_height
        ], dtype=torch.float16, device=self.device)

        # Generate priors (GPU tensors)
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device=self.device, dtype=torch.float16)

        # TensorRT output handling (full GPU tensor flow)
        # Decode boxes directly on GPU tensors
        boxes = decode(loc.squeeze(0), priors, self.cfg['variance'])
        boxes = boxes * scale

        # Decode confidence (GPU tensors)
        scores = conf.squeeze(0)[:, 1]

        # Decode landmarks (GPU tensors)
        landms_decoded = decode_landm(landms.squeeze(0), priors, self.cfg['variance'])
        landms_decoded = landms_decoded * scale1

        # NMS handling (GPU tensors)
        # Filter low confidence
        valid_inds = scores > confidence_threshold
        if not torch.any(valid_inds):
            # Return empty CPU numpy arrays
            empty_dets = np.empty((0, 5), dtype=np.float16)
            empty_landms = np.empty((0, 10), dtype=np.float16)
            return empty_dets, empty_landms

        boxes = boxes[valid_inds]
        landms_decoded = landms_decoded[valid_inds]
        scores = scores[valid_inds]

        # Top-K selection (GPU tensors)
        if len(scores) > top_k:
            _, top_inds = torch.topk(scores, top_k)
            boxes = boxes[top_inds]
            landms_decoded = landms_decoded[top_inds]
            scores = scores[top_inds]

        # NMS (convert to CPU, then keep numpy on CPU)
        dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1)

        # Convert to CPU for NMS (FP16->FP32 for precision)
        dets_cpu = dets.cpu().float().numpy()
        landms_cpu = landms_decoded.cpu().float().numpy()
        keep = py_cpu_nms(dets_cpu, float(nms_threshold))

        # Keep CPU numpy processing
        dets_cpu = dets_cpu[keep]
        landms_cpu = landms_cpu[keep]

        # Final Top-K (CPU numpy)
        if len(dets_cpu) > keep_top_k:
            dets_cpu = dets_cpu[:keep_top_k]
            landms_cpu = landms_cpu[:keep_top_k]

        # Reshape landmarks (CPU numpy)
        landms_cpu = landms_cpu.reshape((-1, 5, 2)).reshape(-1, 10)

        # Final output (CPU numpy, convert to FP16)
        result_dets = (dets_cpu / ss).astype(np.float16)
        result_landms = (landms_cpu / ss).astype(np.float16)

        return result_dets, result_landms

    def __call__(self, *args, **kwargs):
        """Call the model"""
        return self.forward(*args, **kwargs)
