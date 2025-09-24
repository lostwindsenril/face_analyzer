#!/usr/bin/env python3
"""
Face Analyzer unified entry point.
Supports local batch mode and HTTP API server mode.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Config


def setup_loguru():
    """Configure the loguru logging system."""

    from datetime import datetime

    # Remove default handlers
    logger.remove()

    log_level = Config.LOG_LEVEL
    log_mode = Config.LOG_OUTPUT_MODE.lower()

    # If log mode is none, don't add any handlers
    if log_mode == 'none':
        return

    # Set log format
    log_format = "{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}"

    # Add handlers based on config
    if log_mode in ['screen', 'both']:
        logger.add(sys.stdout, format=log_format, level=log_level)

    if log_mode in ['file', 'both']:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"face_analyzer_runtime_{timestamp}.log"
        logger.add(log_file, format=log_format, level=log_level,
                  rotation="10 MB", retention="5 days", encoding="utf-8")


# Initialize loguru
from loguru import logger
setup_loguru()


def setup_local_mode_parser(subparsers):
    """Set up argument parser for local batch mode."""
    local_parser = subparsers.add_parser(
        'local',
        help='本地批处理模式 - 批量处理视频文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
本地批处理模式示例:
  python main.py local input_dir output_dir
  python main.py local /path/to/videos /path/to/output
        """
    )

    local_parser.add_argument(
        'input_base',
        help='输入目录路径，包含待处理的视频文件'
    )
    local_parser.add_argument(
        'output_base',
        help='输出目录路径，处理结果将保存在此目录'
    )

    return local_parser


def setup_server_mode_parser(subparsers):
    """Set up argument parser for HTTP server mode."""
    server_parser = subparsers.add_parser(
        'server',
        help='HTTP API服务器模式 - 启动webp_generator兼容的HTTP服务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HTTP服务器模式示例:
  python main.py server                    # 使用默认配置启动服务器
  python main.py server --host 0.0.0.0    # 指定主机地址
  python main.py server --port 8080       # 指定端口
  python main.py server --debug           # 启用调试模式
        """
    )

    # Import config for defaults
    from config.settings import Config
    default_host = Config.HTTP_HOST
    default_port = Config.HTTP_PORT
    default_debug = Config.HTTP_DEBUG
    default_log_level = Config.LOG_LEVEL

    server_parser.add_argument(
        '--host',
        type=str,
        default=default_host,
        help=f'服务器主机地址 (默认: {default_host})'
    )

    server_parser.add_argument(
        '--port',
        type=int,
        default=default_port,
        help=f'服务器端口 (默认: {default_port})'
    )

    server_parser.add_argument(
        '--debug',
        action='store_true',
        default=default_debug,
        help='启用调试模式'
    )

    server_parser.add_argument(
        '--test-mode',
        action='store_true',
        help='启用测试模式（无需完整依赖）'
    )

    return server_parser


def run_local_mode(args):
    """Run local batch mode."""
    logger.info("启动本地批处理模式...")
    logger.info(f"输入目录: {args.input_base}")
    logger.info(f"输出目录: {args.output_base}")
    logger.info("=" * 60)

    # Import batch manager
    from core.batch_manager import BatchManager

    # Create batch manager and process videos
    processor = BatchManager()
    processor.process_videos(args.input_base, args.output_base)

    logger.info("本地批处理完成")



def print_server_startup_info(host: str, port: int, debug: bool):
    """Print server startup info."""
    from config.settings import Config
    service_name = Config.API_SERVICE_NAME
    api_version = Config.API_VERSION

    logger.info("=" * 60)
    logger.info(f"{service_name} v{api_version}")
    logger.info("=" * 60)
    logger.info(f"服务地址: http://{host}:{port}")
    logger.info(f"调试模式: {'启用' if debug else '禁用'}")
    logger.info(f"兼容模式: webp_generator API")
    logger.info("=" * 60)
    logger.info("可用端点:")
    logger.info("  POST /analyze          - 视频分析和WebP动画生成")
    logger.info("  POST /analyze_frame    - 视频分析和单帧WebP生成")
    logger.info("=" * 60)
    logger.info("服务器启动中...")
def initialize_http_server():
    """Initialize HTTP server."""
    logger.info("正在初始化Face Analyzer HTTP服务器...")

    # Import API manager module directly
    from core.api_manager import create_app

    # Create Flask app
    app = create_app()

    # Configure Flask app
    app.config['MAX_CONTENT_LENGTH'] = Config.HTTP_MAX_CONTENT_LENGTH

    logger.info("服务器初始化完成")
    return app


def run_server_mode(args):
    """Run HTTP server mode."""

    # Print startup info
    print_server_startup_info(args.host, args.port, args.debug)

    # Initialize server
    app = initialize_http_server()

    # Start server
    logger.info(f"启动HTTP服务器: {args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )




def create_main_parser():
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description='Face Analyzer - 人脸分析工具统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用模式:
  local   - 本地批处理模式，批量处理视频文件
  server  - HTTP API服务器模式，提供webp_generator兼容接口

示例:
  # 本地批处理模式
  python main.py local input_dir output_dir

  # HTTP服务器模式
  python main.py server --host 0.0.0.0 --port 8080

  # 查看特定模式的帮助
  python main.py local --help
  python main.py server --help
        """
    )

    # Create subcommand parsers
    subparsers = parser.add_subparsers(
        dest='mode',
        help='运行模式选择',
        metavar='MODE'
    )

    # Set up local and server mode parsers
    setup_local_mode_parser(subparsers)
    setup_server_mode_parser(subparsers)

    return parser


def main():
    """Main function."""
    # Create argument parser
    parser = create_main_parser()

    # Parse CLI args
    args = parser.parse_args()

    # Ensure a mode is specified
    if not args.mode:
        logger.error("错误: 必须指定运行模式")
        parser.print_help()
        sys.exit(1)

    # Run based on mode
    if args.mode == 'local':
        run_local_mode(args)
    elif args.mode == 'server':
        run_server_mode(args)
    else:
        logger.error(f"错误: 未知的运行模式 '{args.mode}'")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
