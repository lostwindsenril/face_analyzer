FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /opt

# Update system and install base tools and build dependencies
RUN apt-get update && apt-get install -y \
    # Base tools
    wget curl git vim unzip \
    # Build tools
    build-essential cmake pkg-config \
    # Python-related
    python3 python3-pip python3-dev python3-numpy-dev \
    # FFmpeg build dependencies
    yasm nasm autoconf automake libtool \
    libx264-dev libx265-dev libvpx-dev \
    libfdk-aac-dev libmp3lame-dev libopus-dev \
    libass-dev libfreetype6-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev \
    libgnutls28-dev libwebp-dev \
    # OpenCV build dependencies
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libavutil-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    libeigen3-dev libgflags-dev libgoogle-glog-dev \
    libhdf5-dev libprotobuf-dev protobuf-compiler \
    libtbb-dev liblapack-dev libopenblas-dev \
    # Additional video processing dependencies
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libdc1394-dev libswresample-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libxine2-dev \
    # Other dependencies
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel numpy

# ============================================================================
# Build from source - NVIDIA Video Codec SDK (ffnvcodec)
# ============================================================================
RUN cd /opt && \
    git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd /opt && rm -rf nv-codec-headers

# ============================================================================
# Build from source - FFmpeg (use local source)
# ============================================================================
COPY ffmpeg /opt/ffmpeg
RUN cd /opt/ffmpeg && \
    ./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include \
                --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared --enable-gpl --enable-gnutls \
                --enable-libass --enable-libfdk-aac --enable-libfreetype --enable-libmp3lame --enable-libopus --enable-libvorbis \
                --enable-libvpx --enable-libx264 --enable-libx265 --enable-nonfree --enable-x86asm --enable-libwebp \
                --extra-cflags='-O3 -march=native' --extra-ldflags=-flto --disable-debug --enable-optimizations && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /opt && rm -rf ffmpeg

# ============================================================================
# Build from source - OpenCV + OpenCV Contrib (use local source)
# ============================================================================
COPY opencv /opt/opencv
COPY opencv_contrib /opt/opencv_contrib
RUN cd /opt/opencv && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D CUDA_ARCH_BIN="7.5,8.0,8.6,8.9" \
        -D WITH_CUBLAS=1 \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D WITH_TBB=ON \
        -D WITH_V4L=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=OFF \
        -D WITH_GSTREAMER=ON \
        -D WITH_FFMPEG=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
        -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
        -D BUILD_opencv_python3=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /opt && rm -rf opencv opencv_contrib

# ============================================================================
# pip install - Decord (use prebuilt package)
# ============================================================================
RUN python3 -m pip install decord

# ============================================================================
# pip install - mv-extractor
# ============================================================================
RUN python3 -m pip install motion-vector-extractor

# ============================================================================
# Install Python dependencies
# ============================================================================

# Install PyTorch (CUDA version)
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Face Analyzer Python dependencies (OpenCV and decord built from source)
RUN python3 -m pip install \
    # Core dependencies
    Pillow \
    loguru \
    flask \
    # Scientific computing
    scipy \
    scikit-learn \
    # Geometry processing
    shapely \
    # Progress bars
    tqdm \
    # Other tools
    requests \
    pathlib \
    typing-extensions



# ============================================================================
# Project setup
# ============================================================================

# Create application directory
WORKDIR /app

# Copy project files
COPY face_analyzer /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create required directories
RUN mkdir -p /app/models/retinaface && \
    mkdir -p /app/models/hopenet && \
    mkdir -p /app/output && \
    mkdir -p /app/cache

# Set permissions
RUN chmod +x /app/main.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Default start command
CMD ["python3", "main.py", "server", "--host", "0.0.0.0", "--port", "5000"]
