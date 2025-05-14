#!/usr/bin/env python3
"""
Process result data class.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class ProcessResult:
    """Video processing result class."""

    # Core fields
    success: bool
    video_path: str
    valid_frames: Dict[float, List[int]]
    error: Optional[str] = field(default=None)

    def __post_init__(self):
        """Post-init processing to ensure valid_frames is a defaultdict."""
        if not isinstance(self.valid_frames, defaultdict):
            # Convert a regular dict to defaultdict if provided.
            temp_dict = defaultdict(list)
            temp_dict.update(self.valid_frames)
            self.valid_frames = temp_dict

    
    def to_dict(self) -> dict:
        """Convert to dict format."""
        result = {
            'video_path': self.video_path,
            'success': self.success,
            'valid_frames': dict(self.valid_frames),
        }
        if self.error:
            result['error'] = self.error
        return result
