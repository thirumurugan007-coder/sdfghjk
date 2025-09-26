"""
Video Player Module for Voice-Controlled CCTV Video Analyzer.

This module provides comprehensive video playback functionality with support for
various video formats, frame navigation, and playback controls.
"""

import cv2
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


class VideoPlayer:
    """
    A comprehensive video player with playback controls and frame management.

    Features:
    - Load and play various video formats
    - Frame-by-frame navigation
    - Playback speed control
    - Seek functionality
    - Frame extraction
    - Video information retrieval
    """

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.frame_delay: float = 1.0 / self.fps
        self.playback_speed: float = 1.0
        self.video_width: int = 0
        self.video_height: int = 0
        self.duration: float = 0.0
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def load_video(self, video_path: str) -> bool:
        """
        Load a video file for playback.

        Args:
            video_path: Path to the video file

        Returns:
            True if video loaded successfully, False otherwise
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Release previous video if loaded
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            return False

        self.video_path = video_path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.current_frame = 0

        return True

    def play(self) -> None:
        """Start video playback in a separate thread."""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("No video loaded")

        if self.is_playing:
            return

        self.is_playing = True
        self.is_paused = False
        self._stop_event.clear()

        self._play_thread = threading.Thread(target=self._play_loop)
        self._play_thread.start()

    def pause(self) -> None:
        """Pause video playback."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume video playback."""
        self.is_paused = False

    def stop(self) -> None:
        """Stop video playback and reset to beginning."""
        self.is_playing = False
        self.is_paused = False
        self._stop_event.set()

        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join()

        self.seek_frame(0)

    def seek_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to (0-based)

        Returns:
            True if seek successful, False otherwise
        """
        if not self.cap or not self.cap.isOpened():
            return False

        frame_number = max(0, min(frame_number, self.total_frames - 1))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number

        return True

    def seek_time(self, seconds: float) -> bool:
        """
        Seek to a specific time in the video.

        Args:
            seconds: Time in seconds to seek to

        Returns:
            True if seek successful, False otherwise
        """
        if not self.cap or not self.cap.isOpened():
            return False

        frame_number = int(seconds * self.fps)
        return self.seek_frame(frame_number)

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame as numpy array.

        Returns:
            Current frame as numpy array, None if error
        """
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get frame at specific frame number.

        Args:
            frame_number: Frame number to retrieve

        Returns:
            Frame as numpy array, None if error
        """
        if not self.cap or not self.cap.isOpened():
            return None

        # Save current position
        current_pos = self.current_frame

        # Seek to desired frame
        if self.seek_frame(frame_number):
            frame = self.get_current_frame()
            # Restore position
            self.seek_frame(current_pos)
            return frame

        return None

    def next_frame(self) -> Optional[np.ndarray]:
        """
        Advance to next frame and return it.

        Returns:
            Next frame as numpy array, None if at end
        """
        if not self.cap or not self.cap.isOpened():
            return None

        if self.current_frame >= self.total_frames - 1:
            return None

        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            return frame

        return None

    def previous_frame(self) -> Optional[np.ndarray]:
        """
        Go to previous frame and return it.

        Returns:
            Previous frame as numpy array, None if at beginning
        """
        if self.current_frame <= 0:
            return None

        target_frame = self.current_frame - 1
        if self.seek_frame(target_frame):
            return self.get_current_frame()

        return None

    def set_playback_speed(self, speed: float) -> None:
        """
        Set playback speed multiplier.

        Args:
            speed: Speed multiplier (1.0 = normal, 2.0 = 2x speed, etc.)
        """
        self.playback_speed = max(0.1, speed)

    def get_video_info(self) -> Dict[str, Any]:
        """
        Get comprehensive video information.

        Returns:
            Dictionary containing video metadata
        """
        if not self.cap or not self.cap.isOpened():
            return {}

        return {
            "path": self.video_path,
            "width": self.video_width,
            "height": self.video_height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "current_frame": self.current_frame,
            "current_time": self.current_frame / self.fps if self.fps > 0 else 0,
            "is_playing": self.is_playing,
            "is_paused": self.is_paused,
            "playback_speed": self.playback_speed,
        }

    def get_current_time(self) -> float:
        """
        Get current playback time in seconds.

        Returns:
            Current time in seconds
        """
        return self.current_frame / self.fps if self.fps > 0 else 0

    def get_progress_percentage(self) -> float:
        """
        Get playback progress as percentage.

        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if self.total_frames <= 0:
            return 0.0
        return (self.current_frame / self.total_frames) * 100.0

    def extract_frames(
        self, start_frame: int, end_frame: int, step: int = 1
    ) -> List[np.ndarray]:
        """
        Extract multiple frames from video.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            step: Step size for frame extraction

        Returns:
            List of extracted frames
        """
        frames = []

        if not self.cap or not self.cap.isOpened():
            return frames

        # Save current position
        current_pos = self.current_frame

        try:
            for frame_num in range(
                start_frame, min(end_frame, self.total_frames), step
            ):
                frame = self.get_frame_at(frame_num)
                if frame is not None:
                    frames.append(frame)
        finally:
            # Restore position
            self.seek_frame(current_pos)

        return frames

    def _play_loop(self) -> None:
        """Internal playback loop running in separate thread."""
        while self.is_playing and not self._stop_event.is_set():
            if self.is_paused:
                time.sleep(0.1)
                continue

            if self.current_frame >= self.total_frames:
                self.is_playing = False
                break

            # Calculate adjusted delay based on playback speed
            adjusted_delay = self.frame_delay / self.playback_speed

            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                break

            self.current_frame += 1

            # Sleep for the appropriate duration
            time.sleep(adjusted_delay)

    def __del__(self):
        """Cleanup resources."""
        self.stop()
        if self.cap:
            self.cap.release()
