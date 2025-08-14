#!/usr/bin/env python3
"""
Logging service module.
Provides unified logging configuration and management.
"""

import os


class ResultStatus:
    """File status logger for success/failure paths."""

    def __init__(self, output_base: str):
        """
        Initialize file status logging.

        Args:
            output_base (str): Output base path
        """
        from config.settings import Config

        self.output_base = output_base

        # Ensure output directory exists
        if not os.path.exists(output_base):
            os.makedirs(output_base, exist_ok=True)

        self.success_log = os.path.join(output_base, Config.RESULT_SUCCESS_LOG)
        self.error_log = os.path.join(output_base, Config.RESULT_ERROR_LOG)
        self.processed_file = os.path.join(output_base, Config.RESULT_PROCESSED_LOG)

    def log_success(self, file_path: str):
        """
        Record a successfully processed file path.

        Args:
            file_path (str): Successfully processed file path
        """
        with open(self.success_log, 'a', encoding='utf-8') as f:
            f.write(file_path + '\n')

    def log_error(self, file_path: str, error_message: str = ""):
        """
        Record a failed file path and error message.

        Args:
            file_path (str): Failed file path
            error_message (str): Error message
        """
        with open(self.error_log, 'a', encoding='utf-8') as f:
            f.write(file_path + '\n')
            f.write(error_message + '\n')

    def get_processed(self):
        """
        Read the processed file list.

        Returns:
            set: Set of processed files
        """
        processed_set = set()
        if os.path.exists(self.processed_file):
            with open(self.processed_file, 'r', encoding='utf-8') as f:
                processed_set = set(line.strip() for line in f if line.strip())
        return processed_set

    def log_processed(self, file_path: str):
        """
        Mark a file as processed.

        Args:
            file_path (str): File path
        """
        with open(self.processed_file, 'a', encoding='utf-8') as f:
            f.write(file_path + '\n')

