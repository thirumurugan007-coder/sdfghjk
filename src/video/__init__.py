"""
Video Processing Module for Voice-Controlled CCTV Video Analyzer

This module provides comprehensive video processing and playback capabilities
specifically designed for CCTV analysis applications.
"""

from .video_player import (
    VideoPlayer,
    VideoMetadata,
    PlaybackState,
    VideoProcessingError,
    create_cctv_player,
    integrate_with_voice_commands,
    demo_video_player,
)

__all__ = [
    "VideoPlayer",
    "VideoMetadata",
    "PlaybackState",
    "VideoProcessingError",
    "create_cctv_player",
    "integrate_with_voice_commands",
    "demo_video_player",
]

__version__ = "1.0.0"
__author__ = "CCTV Video Analyzer Team"
