#!/usr/bin/env python3
"""
File service module.
"""

import os
from config.settings import Config


def collect_video_files(input_base: str):
    """
    Collect video files.

    Args:
        input_base (str): Input base path

    Returns:
        list: Video file paths
    """
    video_paths = []

    if os.path.isfile(input_base):
        # If input is a single file
        if input_base.lower().endswith(tuple(Config.SUPPORTED_VIDEO_EXTENSIONS)):
            video_paths.append(input_base)
    else:
        # If input is a directory
        for root, _, files in os.walk(input_base):
            for fname in files:
                if fname.lower().endswith(tuple(Config.SUPPORTED_VIDEO_EXTENSIONS)):
                    video_paths.append(os.path.join(root, fname))

    return video_paths



def ensure_directory_exists(directory: str):
    """
    Ensure directory exists.

    Args:
        directory (str): Directory path
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_output_path(input_path: str, input_base: str, output_base: str):
    """
    Get output path and base filename.

    Args:
        input_path (str): Input file path
        input_base (str): Input base path
        output_base (str): Output base path

    Returns:
        tuple: (output_dir, base filename without extension)
    """
    fname = os.path.basename(input_path)
    base_name, _ = os.path.splitext(fname)

    # Determine if input_base is a file or directory (by path structure)
    if input_base == input_path or os.path.basename(input_base) == os.path.basename(input_path):
        # If input_base points to a single file, save directly to output dir
        output_dir = output_base
    else:
        # If input_base is a directory, preserve structure
        root = os.path.dirname(input_path)
        rel_dir = os.path.relpath(root, input_base)
        # Handle ".." in relative paths
        if rel_dir == ".":
            output_dir = output_base
        else:
            output_dir = os.path.join(output_base, rel_dir)

    return output_dir, base_name
