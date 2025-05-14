#!/usr/bin/env python3
"""
Frame processing result communication class.
"""

import queue
import threading
from collections import defaultdict
from typing import Dict, List


class FrameProcessingResult:
    """Frame processing result communication class."""

    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=100)  # Frame data queue
        self.stop_event = threading.Event()        # Stop event
        self.valid_frames: Dict[float, List[int]] = defaultdict(list)  # Valid frames grouped by score
        self.lock = threading.Lock()               # Thread lock


# Stop signal constant
STOP_SIGNAL = {'is_stop_signal': True, 'frame': None, 'frame_info': None}
