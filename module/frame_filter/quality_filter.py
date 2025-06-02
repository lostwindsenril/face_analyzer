#!/usr/bin/env python3
"""
Quality frame filter module.
"""

import torch
import cvcuda
from typing import Dict, Any, Tuple
import numpy as np

from .base import BaseFrameFilter
from config.settings import Config


class QualityFrameFilter(BaseFrameFilter):
    """Quality frame filter based on brightness and contrast."""

    def __init__(self, **kwargs):
        """
        Initialize the quality frame filter.

        Args:
            min_brightness (float): Minimum brightness threshold (defaults to config)
            min_contrast (float): Minimum contrast threshold (defaults to config)
            **kwargs: Reserved extension args
        """
        self.filter_name = "QualityFrameFilter"

    def __call__(self, frame: np.ndarray, frame_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply quality filtering.

        Args:
            frame (np.ndarray): Input frame
            frame_info (Dict[str, Any]): Frame info dict

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - is_valid: Whether the frame passes the filter
                - filter_info: Filter debug info dict
        """
        try:
            # Check frame quality
            brightness, contrast, sharpness = self.check_brightness_and_contrast(frame)

            # Convert to floats for comparisons
            brightness_val = float(brightness)
            contrast_val = float(contrast)
            sharpness_val = float(sharpness)
            
            # Evaluate quality pass/fail
            brightness_pass = brightness_val > Config.MIN_BRIGHTNESS
            contrast_pass = contrast_val > Config.MIN_CONTRAST
            overall_pass = brightness_pass and contrast_pass
            
            # Build debug info
            framefilter_info = {
                'brightness': {
                    'value': brightness_val
                },
                'contrast': {
                    'value': contrast_val
                },
                'sharpness': {
                    'value': sharpness_val
                },
            }
            
            # Return results
            return overall_pass, framefilter_info

        except Exception as e:
            # Debug info on exceptions
            framefilter_info = {
                'global_idx': frame_info.get('global_idx', -1),
                'error': str(e)
            }
            return False, framefilter_info


    def check_brightness_and_contrast(self, frame):
        """
        Check image brightness and contrast.

        Args:
            frame (numpy.ndarray): Input image

        Returns:
            tuple: (brightness, contrast_std, sharpness)
        """
        # Convert to HWC layout and ensure contiguous memory
        frame_hwc: torch.Tensor = torch.from_numpy(frame).contiguous().to("cuda")

        # Wrap as CVCUDA tensor with "HWC" layout
        frame_tensor = cvcuda.as_tensor(frame_hwc, layout="HWC")

        gray = cvcuda.cvtcolor(frame_tensor, cvcuda.ColorConversion.BGR2GRAY)

        # Convert to PyTorch GPU tensor and compute mean brightness
        gray_t = torch.as_tensor(gray.cuda()).squeeze(-1).float() * 100.0 / 255.0
        brightness = torch.mean(gray_t)  # Global mean brightness

        # Contrast map: |pixel - brightness|
        contrast_map = torch.abs(gray_t - brightness)
        contrast_std = torch.std(contrast_map)

        # Sharpness estimate: Laplacian + variance
        lap = cvcuda.laplacian(gray, ksize=3, scale=1.0)
        lap_t = torch.as_tensor(lap.cuda()).float()
        sharpness = torch.var(lap_t)  # Variance indicates high-frequency strength

        return brightness, contrast_std, sharpness
