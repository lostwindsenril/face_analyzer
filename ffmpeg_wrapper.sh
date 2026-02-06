#!/bin/bash

# Path to the real ffmpeg executable
REAL_FFMPEG="/usr/local/bin/ffmpeg"

# Collect original arguments
ORIG_ARGS=("$@")

# Log file
LOGFILE="/home/shaohan/video_summerization/test_script/face_analyzer/log-20250418.txt"

# Record input command
echo "[INPUT ] ffmpeg ${ORIG_ARGS[*]}" >> "$LOGFILE"

# Flags
USE_LIBX264=0
USE_LIBWEBP=0
HAS_T_OPTION=0
RESOLUTION_PARAM=""
INPUT_FILE=""
OUTPUT_FILE=""

# Video analysis server configuration
VIDEO_ANALYSIS_HOST="localhost"
VIDEO_ANALYSIS_PORT="5000"
VIDEO_ANALYSIS_TIMEOUT="3600"

# Resolution parsing helpers
parse_vf_scale() {
    local vf_param="$1"
    # Extract scale=WIDTHxHEIGHT or scale=WIDTH:HEIGHT from -vf args
    local scale_pattern="scale=([0-9]+)[x:]([0-9]+)"
    if [[ "$vf_param" =~ $scale_pattern ]]; then
        local width="${BASH_REMATCH[1]}"
        local height="${BASH_REMATCH[2]}"
        echo "${width}x${height}"
        return 0
    fi
    return 1
}

parse_s_param() {
    local s_param="$1"
    # Extract WIDTHxHEIGHT from -s args
    local size_pattern="^([0-9]+)x([0-9]+)$"
    if [[ "$s_param" =~ $size_pattern ]]; then
        echo "$s_param"
        return 0
    fi
    return 1
}

# Retry configuration (hardcoded)
MAX_RETRIES=3
RETRY_INTERVAL=10

# Detect libx264/libwebp usage and extract input/output files
for ((i=0; i<${#ORIG_ARGS[@]}; i++)); do
    case "${ORIG_ARGS[$i]}" in
        -i)
            INPUT_FILE="${ORIG_ARGS[$((i+1))]}"  # First input file
            ;;
        -vcodec|-c:v)
            if [[ "${ORIG_ARGS[$((i+1))]}" == "libx264" ]]; then
                USE_LIBX264=1
            elif [[ "${ORIG_ARGS[$((i+1))]}" == "libwebp" ]]; then
                USE_LIBWEBP=1
            fi
            ;;
        -t)
            HAS_T_OPTION=1
            ;;
        -vf)
            # Parse scale info from -vf args
            vf_value="${ORIG_ARGS[$((i+1))]}"
            if [[ -n "$vf_value" ]]; then
                resolution=$(parse_vf_scale "$vf_value")
                if [[ $? -eq 0 && -n "$resolution" ]]; then
                    RESOLUTION_PARAM="$resolution"
                    echo "[RESOLUTION] 从 -vf 参数捕捉到分辨率: $RESOLUTION_PARAM" >> "$LOGFILE"
                fi
            fi
            ;;
        -s)
            # Parse -s args
            s_value="${ORIG_ARGS[$((i+1))]}"
            if [[ -n "$s_value" ]]; then
                resolution=$(parse_s_param "$s_value")
                if [[ $? -eq 0 && -n "$resolution" ]]; then
                    RESOLUTION_PARAM="$resolution"
                    echo "[RESOLUTION] 从 -s 参数捕捉到分辨率: $RESOLUTION_PARAM" >> "$LOGFILE"
                fi
            fi
            ;;
    esac
done

# Extract output file (usually the last arg)
OUTPUT_FILE="${ORIG_ARGS[-1]}"

# Retry execution function (simple, no color output)
retry_operation() {
    local max_retries="$1"
    local retry_interval="$2"
    local operation_name="$3"
    shift 3
    local command=("$@")

    local attempt=1

    while [ $attempt -le $max_retries ]; do
        if [ $attempt -gt 1 ]; then
            echo "[RETRY] 第 $attempt 次尝试 $operation_name" >> "$LOGFILE"
        else
            echo "[RETRY] 开始 $operation_name" >> "$LOGFILE"
        fi

        # Execute command
        "${command[@]}"
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            if [ $attempt -gt 1 ]; then
                echo "[RETRY] $operation_name 在第 $attempt 次尝试后成功" >> "$LOGFILE"
            fi
            return 0
        fi

        if [ $attempt -lt $max_retries ]; then
            echo "[RETRY] $operation_name 失败 (尝试 $attempt/$max_retries)，${retry_interval}秒后重试" >> "$LOGFILE"
            sleep $retry_interval
        else
            echo "[RETRY] $operation_name 在 $max_retries 次尝试后仍然失败" >> "$LOGFILE"
        fi

        ((attempt++))
    done

    return 1
}

# Video analysis functions
# Send video analysis request to the server (single attempt)
send_video_analysis_request_single() {
    local input_path="$1"
    local output_path="$2"
    local resolution="$3"
    local url="http://${VIDEO_ANALYSIS_HOST}:${VIDEO_ANALYSIS_PORT}/analyze"

    # Resolve absolute paths
    input_path=$(realpath "$input_path" 2>/dev/null || echo "$input_path")
    output_path=$(realpath "$output_path" 2>/dev/null || echo "$output_path")

    # Build JSON request body
    if [[ -n "$resolution" ]]; then
        json_data=$(cat <<EOF
{
    "input_path": "$input_path",
    "output_path": "$output_path",
    "resolution": "$resolution"
}
EOF
)
    else
        json_data=$(cat <<EOF
{
    "input_path": "$input_path",
    "output_path": "$output_path"
}
EOF
)
    fi

    # Send HTTP POST request
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_data" \
        --connect-timeout 30 \
        --max-time "$VIDEO_ANALYSIS_TIMEOUT" \
        "$url" 2>/dev/null)

    # Check curl execution result
    if [ $? -ne 0 ]; then
        echo "[VIDEO_ANALYSIS] 请求失败：$input_path" >> "$LOGFILE"
        return 1
    fi

    # Parse response
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    if [ "$http_code" = "200" ]; then
        echo "[VIDEO_ANALYSIS] 视频分析完成" >> "$LOGFILE"

        # Check whether output file was created
        if [ -f "$output_path" ]; then
            file_size=$(du -h "$output_path" 2>/dev/null | cut -f1 || echo "未知")
            echo "[VIDEO_ANALYSIS] 输出文件已生成: $output_path (大小: $file_size)" >> "$LOGFILE"
            return 0
        else
            echo "[VIDEO_ANALYSIS] 输出文件未找到: $output_path" >> "$LOGFILE"
            return 1
        fi
    else
        echo "[VIDEO_ANALYSIS] 服务器返回错误状态码: $http_code" >> "$LOGFILE"
        echo "[VIDEO_ANALYSIS] 错误详情: $body" >> "$LOGFILE"
        return 1
    fi
}

# Send frame analysis request to the server (single attempt)
send_frame_analysis_request_single() {
    local input_path="$1"
    local output_path="$2"
    local resolution="$3"
    local url="http://${VIDEO_ANALYSIS_HOST}:${VIDEO_ANALYSIS_PORT}/analyze_frame"

    # Resolve absolute paths
    input_path=$(realpath "$input_path" 2>/dev/null || echo "$input_path")
    output_path=$(realpath "$output_path" 2>/dev/null || echo "$output_path")

    # Build JSON request body
    if [[ -n "$resolution" ]]; then
        json_data=$(cat <<EOF
{
    "input_path": "$input_path",
    "output_path": "$output_path",
    "resolution": "$resolution"
}
EOF
)
    else
        json_data=$(cat <<EOF
{
    "input_path": "$input_path",
    "output_path": "$output_path"
}
EOF
)
    fi

    # Send HTTP POST request
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_data" \
        --connect-timeout 30 \
        --max-time "$VIDEO_ANALYSIS_TIMEOUT" \
        "$url" 2>/dev/null)

    # Check curl execution result
    if [ $? -ne 0 ]; then
        echo "[FRAME_ANALYSIS] 请求失败：$input_path" >> "$LOGFILE"
        return 1
    fi

    # Parse response
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    if [ "$http_code" = "200" ]; then
        echo "[FRAME_ANALYSIS] 单帧分析完成" >> "$LOGFILE"

        # Check whether output file was created
        if [ -f "$output_path" ]; then
            file_size=$(du -h "$output_path" 2>/dev/null | cut -f1 || echo "未知")
            echo "[FRAME_ANALYSIS] 输出文件已生成: $output_path (大小: $file_size)" >> "$LOGFILE"
            return 0
        else
            echo "[FRAME_ANALYSIS] 输出文件未找到: $output_path" >> "$LOGFILE"
            return 1
        fi
    else
        echo "[FRAME_ANALYSIS] 服务器返回错误状态码: $http_code" >> "$LOGFILE"
        echo "[FRAME_ANALYSIS] 错误详情: $body" >> "$LOGFILE"
        return 1
    fi
}

# Send video analysis request with retries
send_video_analysis_request() {
    local input_path="$1"
    local output_path="$2"
    local resolution="$3"

    # Validate input file
    if [ ! -f "$input_path" ]; then
        echo "[VIDEO_ANALYSIS] 错误: 输入文件不存在: $input_path" >> "$LOGFILE"
        return 1
    fi

    if [[ -n "$resolution" ]]; then
        echo "[VIDEO_ANALYSIS] 开始视频分析: $input_path -> $output_path (分辨率: $resolution)" >> "$LOGFILE"
    else
        echo "[VIDEO_ANALYSIS] 开始视频分析: $input_path -> $output_path" >> "$LOGFILE"
    fi
    echo "[VIDEO_ANALYSIS] 发送请求到: http://${VIDEO_ANALYSIS_HOST}:${VIDEO_ANALYSIS_PORT}/analyze" >> "$LOGFILE"

    # Use retry mechanism
    retry_operation "$MAX_RETRIES" "$RETRY_INTERVAL" "视频分析" send_video_analysis_request_single "$input_path" "$output_path" "$resolution"
    return $?
}

# Send frame analysis request with retries
send_frame_analysis_request() {
    local input_path="$1"
    local output_path="$2"
    local resolution="$3"

    # Validate input file
    if [ ! -f "$input_path" ]; then
        echo "[FRAME_ANALYSIS] 错误: 输入文件不存在: $input_path" >> "$LOGFILE"
        return 1
    fi

    if [[ -n "$resolution" ]]; then
        echo "[FRAME_ANALYSIS] 开始单帧分析: $input_path -> $output_path (分辨率: $resolution)" >> "$LOGFILE"
    else
        echo "[FRAME_ANALYSIS] 开始单帧分析: $input_path -> $output_path" >> "$LOGFILE"
    fi
    echo "[FRAME_ANALYSIS] 发送请求到: http://${VIDEO_ANALYSIS_HOST}:${VIDEO_ANALYSIS_PORT}/analyze_frame" >> "$LOGFILE"

    # Use retry mechanism
    retry_operation "$MAX_RETRIES" "$RETRY_INTERVAL" "单帧分析" send_frame_analysis_request_single "$input_path" "$output_path" "$resolution"
    return $?
}

# Split arguments into input and output parts
INPUT_PART=()
OUTPUT_PART=()
last_i=-1
for i in "${!ORIG_ARGS[@]}"; do
    if [[ "${ORIG_ARGS[$i]}" == "-i" ]]; then
        last_i=$i
    fi
done
# Include "-i" and its filename
end_index=$((last_i + 1))
for ((i=0; i<=end_index; i++)); do
    INPUT_PART+=("${ORIG_ARGS[$i]}")
done
# The rest are output options and output file
for ((i=end_index+1; i<${#ORIG_ARGS[@]}; i++)); do
    OUTPUT_PART+=("${ORIG_ARGS[$i]}")
done

# Reassemble args: input part first
NEW_ARGS=("${INPUT_PART[@]}")

# Insert libx264 GOP control params after inputs
if [[ $USE_LIBX264 -eq 1 ]]; then
    NEW_ARGS+=("-g" "30" "-keyint_min" "30" "-sc_threshold" "0")
fi

# OUTPUT_PART is already populated; last element is output filename
OUT_FILE="${OUTPUT_PART[-1]}"

# If output ends with .webp (case-insensitive), treat as libwebp
if [[ "${OUT_FILE,,}" == *.webp ]]; then
    USE_LIBWEBP=1
fi

# New USE_LIBWEBP flow: integrate video analysis
if [[ $USE_LIBWEBP -eq 1 && -n "$INPUT_FILE" && -n "$OUTPUT_FILE" ]]; then
    # Choose mode based on presence of -t option
    if [[ $HAS_T_OPTION -eq 1 ]]; then
        echo "[WEBP_PROCESSING] 检测到 libwebp 使用和 -t 选项，尝试视频分析服务器（动画模式）" >> "$LOGFILE"
        analysis_mode="动画"
        analysis_function="send_video_analysis_request"
    else
        echo "[WEBP_PROCESSING] 检测到 libwebp 使用，无 -t 选项，尝试单帧分析服务器（单帧模式）" >> "$LOGFILE"
        analysis_mode="单帧"
        analysis_function="send_frame_analysis_request"
    fi

    # Ensure output file ends with .webp
    if [[ ! "${OUTPUT_FILE,,}" == *.webp ]]; then
        OUTPUT_FILE="${OUTPUT_FILE}.webp"
        echo "[WEBP_PROCESSING] 自动添加 .webp 扩展名: $OUTPUT_FILE" >> "$LOGFILE"
    fi

    # Create output directory
    output_dir=$(dirname "$OUTPUT_FILE")
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "[WEBP_PROCESSING] 已创建输出目录: $output_dir" >> "$LOGFILE"
        else
            echo "[WEBP_PROCESSING] 警告: 无法创建输出目录: $output_dir" >> "$LOGFILE"
        fi
    fi

    # Try the appropriate analysis server
    if $analysis_function "$INPUT_FILE" "$OUTPUT_FILE" "$RESOLUTION_PARAM"; then
        echo "[WEBP_PROCESSING] ${analysis_mode}分析服务器处理成功，跳过 FFmpeg 处理" >> "$LOGFILE"
        exit 0
    else
        echo "[WEBP_PROCESSING] ${analysis_mode}分析服务器处理失败，回退到 FFmpeg 处理" >> "$LOGFILE"
        # Continue with original FFmpeg logic
        DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT_FILE" 2>/dev/null)
        if [[ -n "$DURATION" ]]; then
            echo "[DURATION] $INPUT_FILE: ${DURATION}s" >> "$LOGFILE"
            D_INT=${DURATION%%.*}
            if (( D_INT > 30 )); then
                NEW_ARGS+=("-ss" "30")
            fi
        fi
    fi
fi

# Append output args and output file
NEW_ARGS+=("${OUTPUT_PART[@]}")

# Final command
FINAL_CMD=("$REAL_FFMPEG" "${NEW_ARGS[@]}")
# Record output command
echo "[OUTPUT] ${FINAL_CMD[*]}" >> "$LOGFILE"

# Execute the real ffmpeg
exec "${FINAL_CMD[@]}"
