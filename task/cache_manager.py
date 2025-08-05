#!/usr/bin/env python3
"""
Face analyzer cache manager - TinyDB refactor.
"""

import os
import time
import hashlib
import cv2
import threading
import atexit
from typing import Optional, Dict, Any, List, Union
from collections import defaultdict

from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage
from loguru import logger

from config.settings import Config
from classes.process_result import ProcessResult


class FaceAnalyzerCacheManager:
    """Face analyzer cache manager - TinyDB refactor."""

    def __init__(self, flush_interval: float = 8.0):
        """
        Initialize the cache manager.

        Args:
            flush_interval (float): Async flush interval (seconds), default 8s
        """

        # TinyDB instance - use existing JSON file directly.
        self.db = TinyDB(Config.CACHE_FILE)
        self.cache_table = self.db.table('cache')

        # In-memory cache for fast access, JSON-compatible format.
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.dirty_keys: set = set()  # Keys that need persistence

        # Async flush configuration
        self.flush_interval = flush_interval
        self.flush_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Start async flush thread
        self._start_flush_thread()

        # Register cleanup on exit
        atexit.register(self._cleanup)

        # Load existing data into memory on init
        self._load_existing_data()

        logger.info(f"TinyDB缓存管理器初始化完成，JSON文件: {Config.CACHE_FILE}")

    def _start_flush_thread(self):
        """Start the async flush thread."""
        if self.flush_thread is None or not self.flush_thread.is_alive():
            self.flush_thread = threading.Thread(
                target=self._flush_worker,
                name="CacheFlushThread",
                daemon=True
            )
            self.flush_thread.start()
            logger.debug("异步刷盘线程已启动")

    def _flush_worker(self):
        """Async flush worker thread."""
        while not self.stop_event.is_set():
            try:
                # Wait for flush interval or stop signal
                if self.stop_event.wait(timeout=self.flush_interval):
                    break

                # Perform flush
                self._flush_to_disk()

            except Exception as e:
                logger.error(f"异步刷盘线程异常: {e}")

    def _flush_to_disk(self):
        """Flush dirty in-memory data to disk using the JSON file."""
        if not self.dirty_keys:
            return

        # Copy then clear dirty keys (single-threaded, no lock needed).
        dirty_keys_copy = self.dirty_keys.copy()
        self.dirty_keys.clear()

        if not dirty_keys_copy:
            return

        try:
            import json
            import shutil

            # Build full cache structure, compatible with original JSON format.
            cache_data = {"cache": {}}

            # Build JSON from in-memory cache.
            for signature, cache_entry in self.memory_cache.items():
                # Remove signature to match original JSON format.
                clean_entry = {k: v for k, v in cache_entry.items() if k != 'signature'}
                cache_data["cache"][signature] = clean_entry

            # Safe write using a temp file.
            temp_file = Config.CACHE_FILE + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            # Atomic replace
            shutil.move(temp_file, Config.CACHE_FILE)

            updated_count = len(dirty_keys_copy)
            if updated_count > 0:
                logger.debug(f"异步刷盘完成，更新了 {updated_count} 条记录")

        except Exception as e:
            logger.error(f"刷盘操作失败: {e}")
            # Re-mark failed keys as dirty (single-threaded, no lock needed).
            self.dirty_keys.update(dirty_keys_copy)

    def _load_existing_data(self):
        """Load existing JSON data into the in-memory cache."""
        try:
            # Load directly from JSON to avoid TinyDB type conversion issues.
            if os.path.exists(Config.CACHE_FILE):
                import json
                with open(Config.CACHE_FILE, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                loaded_count = 0
                if "cache" in json_data:
                    cache_entries = json_data["cache"]
                    for signature, cache_data in cache_entries.items():
                        # Add signature field for in-memory indexing.
                        memory_entry = cache_data.copy()
                        memory_entry["signature"] = signature
                        self.memory_cache[signature] = memory_entry
                        loaded_count += 1

                logger.info(f"加载了 {loaded_count} 条缓存记录到内存")
            else:
                logger.info("缓存文件不存在，从空缓存开始")

        except Exception as e:
            logger.error(f"加载现有数据失败: {e}")

    def _cleanup(self):
        """Clean up resources and ensure data persistence."""
        logger.info("开始清理缓存管理器资源...")

        # Stop flush thread
        self.stop_event.set()

        # Final flush
        self._flush_to_disk()

        # Wait for flush thread to finish
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)

        # Close database
        if hasattr(self, 'db'):
            self.db.close()

        logger.info("缓存管理器资源清理完成")

    def get_file_signature(self, file_path: str) -> str:
        """Generate an MD5 signature from path, size, and mtime."""
        abs_path = os.path.abspath(file_path)
        stat = os.stat(abs_path)
        signature_data = f"{abs_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(signature_data.encode()).hexdigest()

    def get_cached_result(self, file_path: str) -> Optional[Dict]:
        """Get cached result (prefer in-memory cache)."""
        file_signature = self.get_file_signature(file_path)

        # Prefer in-memory cache
        cached_info = self.memory_cache.get(file_signature)

        if cached_info:
            # Ensure file still exists and hasn't changed
            if os.path.exists(file_path):
                current_signature = self.get_file_signature(file_path)
                if current_signature == file_signature:
                    logger.debug(f"缓存命中（内存）: {file_path}")
                    return {
                        "file_path": cached_info["file_path"],
                        "valid_frames": cached_info["valid_frames"],
                        "cached_at": cached_info["cached_at"]
                    }
                else:
                    # File changed; remove stale cache
                    self._remove_cache_entry(file_signature)

        logger.debug(f"缓存未命中: {file_path}")
        return None

    def cache_result(self, file_path: str, score_groups: Dict[float, List[int]]) -> None:
        """Cache result in memory while keeping JSON compatibility."""
        file_signature = self.get_file_signature(file_path)

        # Create new cache entry matching original JSON format.
        valid_frames_str = {str(score): sorted(indices) for score, indices in score_groups.items()}
        cache_entry = {
            "signature": file_signature,  # Keep signature for in-memory indexing
            "file_path": os.path.abspath(file_path),
            "valid_frames": valid_frames_str,
            "cached_at": time.time()
        }

        # Store in memory
        self.memory_cache[file_signature] = cache_entry

        # Mark dirty for async persistence (single-threaded, no lock needed).
        self.dirty_keys.add(file_signature)

        logger.debug(f"缓存已更新（内存）: {file_path}")

    def _remove_cache_entry(self, signature: str):
        """Remove a cache entry while keeping JSON compatibility."""
        # Remove from memory cache
        if signature in self.memory_cache:
            del self.memory_cache[signature]

        # Mark dirty so flush updates the full JSON structure (single-threaded).
        self.dirty_keys.add(signature)  # Mark as dirty; flush handles deletion

        logger.debug(f"缓存条目已标记删除: {signature}")
    
    def clear_cache(self) -> bool:
        """Clear all caches using direct JSON file operations."""
        try:
            import json

            # Clear in-memory cache
            self.memory_cache.clear()

            # Clear dirty marks (single-threaded, no lock needed).
            self.dirty_keys.clear()

            # Write an empty JSON structure
            empty_cache = {"cache": {}}
            with open(Config.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(empty_cache, f, ensure_ascii=False, indent=2)

            logger.info("缓存已清空")
            return True

        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            json_file = Config.CACHE_FILE
            json_size = os.path.getsize(json_file) if os.path.exists(json_file) else 0

            # Count entries directly from JSON file
            json_cache_entries = 0
            if os.path.exists(json_file):
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if "cache" in json_data:
                        json_cache_entries = len(json_data["cache"])

            memory_entries = len(self.memory_cache)
            dirty_entries = len(self.dirty_keys)

            return {
                "json_file": json_file,
                "json_cache_entries": json_cache_entries,
                "memory_entries": memory_entries,
                "dirty_entries": dirty_entries,
                "json_exists": os.path.exists(json_file),
                "json_size_bytes": json_size,
                "flush_interval": self.flush_interval,
                "flush_thread_alive": self.flush_thread.is_alive() if self.flush_thread else False
            }

        except Exception as e:
            logger.error(f"获取缓存统计信息失败: {e}")
            return {
                "error": str(e),
                "json_file": Config.CACHE_FILE,
                "json_cache_entries": 0,
                "memory_entries": len(self.memory_cache),
                "dirty_entries": len(self.dirty_keys),
                "json_exists": os.path.exists(Config.CACHE_FILE),
                "json_size_bytes": 0,
                "flush_interval": self.flush_interval,
                "flush_thread_alive": self.flush_thread.is_alive() if self.flush_thread else False
            }

    def force_flush(self) -> bool:
        """Force an immediate flush to disk."""
        try:
            self._flush_to_disk()
            logger.info("强制刷盘完成")
            return True
        except Exception as e:
            logger.error(f"强制刷盘失败: {e}")
            return False


def process_with_cache(input_path: str, cached_data: dict) -> ProcessResult:
    """Process a video using cached data."""

    valid_frames_str = cached_data['valid_frames']
    # Convert string keys back to floats
    valid_frames = {float(score): indices for score, indices in valid_frames_str.items()}

    return ProcessResult(
        success=True,
        video_path=input_path,
        valid_frames=valid_frames
    )





def extract_and_save_cached_frame(video_path: str, output_path: str, frame_idx: Union[int, List[int]], face_processor) -> bool:
    """Extract and save cached frames (single or multiple indices)."""
    try:
        # Ensure frame_idx is list-like
        if isinstance(frame_idx, int):
            frame_idx_list = [frame_idx]
        else:
            frame_idx_list = frame_idx

        # Jump directly to the target frame index using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return False

        success_count = 0
        for idx in frame_idx_list:
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # Read frame
            ret, frame = cap.read()

            if ret:
                # Save frame (first successful frame is the primary result)
                if success_count == 0:
                    face_processor.result_handler.save_result(
                        frame, output_path, 0, is_final=True, frame_idx=idx
                    )
                success_count += 1

        cap.release()
        logger.debug(f"成功提取并保存了 {success_count} 帧")
        return success_count > 0

    except Exception as e:
        logger.error(f"提取并保存缓存帧失败: {e}")
        return False
