#!/usr/bin/env python3
"""
API manager module.
Defines all Flask routes and endpoints.
Focuses on HTTP layer handling: routing, request parsing, response formatting.
"""

from flask import Flask, request, jsonify

from task.cache_manager import FaceAnalyzerCacheManager
from core.api_helper import analyze_and_generate_webp_animation, analyze_and_generate_single_webp
from utils.concurrency_control import ConcurrencyController, concurrency_limit
from config.settings import Config

from loguru import logger

def create_app() -> Flask:
    """
    Create a Flask app instance.

    Returns:
        Flask: Configured Flask app instance
    """
    app = Flask(__name__)

    # Initialize core components
    cache_manager = FaceAnalyzerCacheManager()

    # Initialize concurrency controller
    concurrency_controller = ConcurrencyController(Config.CONCURRENCY_LIMIT)

    @app.route('/analyze', methods=['POST'])
    @concurrency_limit(concurrency_controller)
    def analyze_video():
        """Video analysis API endpoint."""
        # Parse and validate JSON request
        data = request.get_json()

        input_path = data.get('input_path')
        output_path = data.get('output_path')
        resolution = data.get('resolution')

        # Log request start
        log_msg = f"开始处理请求: {input_path} -> {output_path}"
        if resolution:
            log_msg += f", 分辨率: {resolution}"
        logger.info(log_msg)

        # Call business logic
        result = analyze_and_generate_webp_animation(
            cache_manager, input_path, output_path, resolution
        )

        if result.get("success", True):
            logger.info(f"处理完成: {result}")
            return jsonify(result)
        else:
            logger.error(f"处理失败: {result}")
            return jsonify(result), 500

    @app.route('/analyze_frame', methods=['POST'])
    @concurrency_limit(concurrency_controller)
    def analyze_video_frame():
        """Single-frame analysis API endpoint."""
        # Parse and validate JSON request
        data = request.get_json()

        input_path = data.get('input_path')
        output_path = data.get('output_path')
        resolution = data.get('resolution')

        # Log request start
        log_msg = f"开始处理单帧请求: {input_path} -> {output_path}"
        if resolution:
            log_msg += f", 分辨率: {resolution}"
        logger.info(log_msg)

        # Call business logic
        result = analyze_and_generate_single_webp(
            cache_manager, input_path, output_path, resolution
        )

        if result.get("success", True):
            logger.info(f"单帧处理完成: {result}")
            return jsonify(result)
        else:
            logger.error(f"单帧处理失败: {result}")
            return jsonify(result), 500

    return app
