#!/usr/bin/env python3
"""
Concurrency control module.
"""

import threading
from functools import wraps

from loguru import logger


class ConcurrencyController:
    """Concurrency controller."""
    
    def __init__(self, max_concurrent: int):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_requests = 0
        self.lock = threading.Lock()
    
    def acquire(self):
        self.semaphore.acquire()
        with self.lock:
            self.active_requests += 1
    
    def release(self):
        with self.lock:
            self.active_requests -= 1
        self.semaphore.release()
    
    def get_active_count(self):
        with self.lock:
            return self.active_requests


def concurrency_limit(controller):
    """Concurrency limit decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"请求开始，当前活跃请求数: {controller.get_active_count()}")
            
            controller.acquire()
            
            try:
                logger.info(f"请求开始处理，当前活跃请求数: {controller.get_active_count()}")
                result = func(*args, **kwargs)
                return result
            finally:
                controller.release()
                logger.info(f"请求处理完成，当前活跃请求数: {controller.get_active_count()}")
        
        return wrapper
    return decorator
