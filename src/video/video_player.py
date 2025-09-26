"""
Complete Video Player Module for Voice-Controlled CCTV Video Analyzer

This module provides comprehensive video processing functionality including:
- Video file loading and playback controls
- Frame extraction and processing
- Format conversion and scaling
- Integration with YOLO detection and voice commands
- CCTV-specific features like event logging and timestamp management
"""

import cv2
import time
import threading
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pygame
import logging

# Optional dependencies with fallbacks
try:
    import ffmpeg

    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

try:
    from moviepy.editor import VideoFileClip

    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Container for video metadata"""

    filename: str
    duration: float
    fps: float
    frame_count: int
    width: int
    height: int
    format: str
    size_bytes: int
    codec: str
    bitrate: Optional[int] = None


@dataclass
class PlaybackState:
    """Container for current playback state"""

    is_playing: bool = False
    is_paused: bool = False
    current_frame: int = 0
    current_time: float = 0.0
    volume: float = 1.0
    playback_speed: float = 1.0


class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""

    pass


class VideoPlayer:
    """
    Complete video player with comprehensive video processing functionality
    designed for CCTV analysis and voice-controlled operations.
    """

    SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}

    def __init__(
        self,
        display_size: Tuple[int, int] = (800, 600),
        enable_audio: bool = True,
        enable_logging: bool = True,
    ):
        """
        Initialize the video player

        Args:
            display_size: Target display resolution (width, height)
            enable_audio: Whether to enable audio playback
            enable_logging: Whether to enable event logging
        """
        self.display_size = display_size
        self.enable_audio = enable_audio
        self.enable_logging = enable_logging

        # Video capture and processing
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_clip: Optional[VideoFileClip] = None
        self.current_video_path: Optional[str] = None

        # State management
        self.metadata: Optional[VideoMetadata] = None
        self.playback_state = PlaybackState()
        self.frame_buffer: List[np.ndarray] = []

        # Threading for smooth playback
        self.playback_thread: Optional[threading.Thread] = None
        self.playback_lock = threading.Lock()
        self._stop_playback = threading.Event()

        # Event callbacks
        self.frame_callbacks: List[Callable[[np.ndarray, int], None]] = []
        self.event_callbacks: List[Callable[[str, Dict], None]] = []

        # Initialize pygame for audio (if enabled)
        if self.enable_audio:
            try:
                pygame.mixer.init()
                logger.info("Audio system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize audio: {e}")
                self.enable_audio = False

        logger.info("VideoPlayer initialized successfully")

    def load_video(self, video_path: str) -> bool:
        """
        Load a video file for processing and playback

        Args:
            video_path: Path to the video file

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            video_path = Path(video_path).resolve()

            if not video_path.exists():
                raise VideoProcessingError(f"Video file not found: {video_path}")

            if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise VideoProcessingError(
                    f"Unsupported video format: {video_path.suffix}"
                )

            # Clean up previous video
            self.close_video()

            # Load with OpenCV for frame processing
            self.cap = cv2.VideoCapture(str(video_path))
            if not self.cap.isOpened():
                raise VideoProcessingError("Failed to open video with OpenCV")

            # Load with MoviePy for audio and advanced operations
            if self.enable_audio and HAS_MOVIEPY:
                try:
                    self.video_clip = VideoFileClip(str(video_path))
                except Exception as e:
                    logger.warning(f"Failed to load audio track: {e}")
                    self.enable_audio = False
            elif self.enable_audio:
                logger.warning("MoviePy not available, disabling audio features")
                self.enable_audio = False

            # Extract metadata
            self.current_video_path = str(video_path)
            self.metadata = self._extract_metadata(video_path)

            # Reset playback state
            self.playback_state = PlaybackState()

            # Pre-load first frame
            self._load_frame_buffer(0, min(30, self.metadata.frame_count))

            self._log_event(
                "video_loaded", {"path": str(video_path), "metadata": self.metadata}
            )
            logger.info(f"Video loaded successfully: {video_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {e}")
            return False

    def play(self) -> bool:
        """
        Start or resume video playback

        Returns:
            bool: True if playback started successfully
        """
        if not self.cap or not self.metadata:
            logger.error("No video loaded")
            return False

        with self.playback_lock:
            if self.playback_state.is_playing:
                logger.info("Video is already playing")
                return True

            self.playback_state.is_playing = True
            self.playback_state.is_paused = False
            self._stop_playback.clear()

            # Start playback thread
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(
                    target=self._playback_loop, daemon=True
                )
                self.playback_thread.start()

            self._log_event(
                "playback_started", {"frame": self.playback_state.current_frame}
            )
            logger.info("Video playback started")
            return True

    def pause(self) -> bool:
        """
        Pause video playback

        Returns:
            bool: True if paused successfully
        """
        with self.playback_lock:
            if not self.playback_state.is_playing:
                logger.info("Video is not playing")
                return False

            self.playback_state.is_paused = True
            self._log_event(
                "playback_paused", {"frame": self.playback_state.current_frame}
            )
            logger.info("Video playback paused")
            return True

    def stop(self) -> bool:
        """
        Stop video playback and return to beginning

        Returns:
            bool: True if stopped successfully
        """
        with self.playback_lock:
            self.playback_state.is_playing = False
            self.playback_state.is_paused = False
            self._stop_playback.set()

            # Reset to beginning
            self.seek_to_frame(0)

            self._log_event("playback_stopped", {})
            logger.info("Video playback stopped")
            return True

    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame

        Args:
            frame_number: Target frame number

        Returns:
            bool: True if seek successful
        """
        if not self.cap or not self.metadata:
            return False

        frame_number = max(0, min(frame_number, self.metadata.frame_count - 1))

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.playback_state.current_frame = frame_number
            self.playback_state.current_time = frame_number / self.metadata.fps

            self._log_event("seek_performed", {"frame": frame_number})
            logger.info(f"Seeked to frame {frame_number}")
            return True

        except Exception as e:
            logger.error(f"Failed to seek to frame {frame_number}: {e}")
            return False

    def seek_to_time(self, time_seconds: float) -> bool:
        """
        Seek to a specific time position

        Args:
            time_seconds: Target time in seconds

        Returns:
            bool: True if seek successful
        """
        if not self.metadata:
            return False

        target_frame = int(time_seconds * self.metadata.fps)
        return self.seek_to_frame(target_frame)

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame as numpy array

        Returns:
            Current frame or None if not available
        """
        if not self.cap:
            return None

        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
        except Exception as e:
            logger.error(f"Failed to get current frame: {e}")
            return None

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number

        Args:
            frame_number: Target frame number

        Returns:
            Frame as numpy array or None if not available
        """
        if not self.cap or not self.metadata:
            return None

        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        try:
            # Seek to target frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()

            # Restore original position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

            return frame if ret else None

        except Exception as e:
            logger.error(f"Failed to get frame at {frame_number}: {e}")
            return None

    def extract_frames(
        self, start_frame: int = 0, end_frame: Optional[int] = None, step: int = 1
    ) -> List[np.ndarray]:
        """
        Extract a sequence of frames

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (None for end of video)
            step: Frame step size

        Returns:
            List of frames as numpy arrays
        """
        if not self.cap or not self.metadata:
            return []

        end_frame = end_frame or self.metadata.frame_count - 1
        frames = []

        try:
            for frame_num in range(
                start_frame, min(end_frame + 1, self.metadata.frame_count), step
            ):
                frame = self.get_frame_at(frame_num)
                if frame is not None:
                    frames.append(frame)

            logger.info(f"Extracted {len(frames)} frames")
            return frames

        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            return frames

    def generate_thumbnail(
        self,
        time_seconds: float = None,
        frame_number: int = None,
        size: Tuple[int, int] = (200, 150),
    ) -> Optional[np.ndarray]:
        """
        Generate a thumbnail from the video

        Args:
            time_seconds: Time position for thumbnail (takes precedence)
            frame_number: Frame number for thumbnail
            size: Thumbnail size (width, height)

        Returns:
            Thumbnail as numpy array or None
        """
        if time_seconds is not None:
            frame_number = int(time_seconds * self.metadata.fps) if self.metadata else 0
        elif frame_number is None:
            frame_number = self.metadata.frame_count // 2 if self.metadata else 0

        frame = self.get_frame_at(frame_number)
        if frame is None:
            return None

        try:
            thumbnail = cv2.resize(frame, size)
            logger.info(f"Generated thumbnail from frame {frame_number}")
            return thumbnail
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            return None

    def apply_frame_filter(
        self, frame: np.ndarray, filter_type: str = "none", **kwargs
    ) -> np.ndarray:
        """
        Apply image processing filters to a frame

        Args:
            frame: Input frame
            filter_type: Type of filter to apply
            **kwargs: Additional filter parameters

        Returns:
            Processed frame
        """
        try:
            if filter_type == "grayscale":
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif filter_type == "blur":
                kernel_size = kwargs.get("kernel_size", 5)
                return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            elif filter_type == "sharpen":
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                return cv2.filter2D(frame, -1, kernel)
            elif filter_type == "edge":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            elif filter_type == "enhance":
                # Simple contrast and brightness enhancement
                alpha = kwargs.get("contrast", 1.2)
                beta = kwargs.get("brightness", 10)
                return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            else:
                return frame

        except Exception as e:
            logger.error(f"Failed to apply filter {filter_type}: {e}")
            return frame

    def set_playback_speed(self, speed: float) -> bool:
        """
        Set playback speed multiplier

        Args:
            speed: Speed multiplier (1.0 = normal, 2.0 = double speed, etc.)

        Returns:
            bool: True if set successfully
        """
        if speed <= 0:
            return False

        with self.playback_lock:
            self.playback_state.playback_speed = speed
            self._log_event("speed_changed", {"speed": speed})
            logger.info(f"Playback speed set to {speed}x")
            return True

    def set_volume(self, volume: float) -> bool:
        """
        Set audio volume

        Args:
            volume: Volume level (0.0 to 1.0)

        Returns:
            bool: True if set successfully
        """
        volume = max(0.0, min(1.0, volume))

        with self.playback_lock:
            self.playback_state.volume = volume

            if self.enable_audio and pygame.mixer.get_init():
                pygame.mixer.music.set_volume(volume)

            self._log_event("volume_changed", {"volume": volume})
            logger.info(f"Volume set to {volume}")
            return True

    def add_frame_callback(self, callback: Callable[[np.ndarray, int], None]) -> None:
        """
        Add a callback function that will be called for each frame during playback

        Args:
            callback: Function that takes (frame, frame_number) as arguments
        """
        self.frame_callbacks.append(callback)
        logger.info("Frame callback added")

    def add_event_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """
        Add a callback for video player events

        Args:
            callback: Function that takes (event_type, event_data) as arguments
        """
        self.event_callbacks.append(callback)
        logger.info("Event callback added")

    def get_metadata(self) -> Optional[VideoMetadata]:
        """
        Get video metadata

        Returns:
            VideoMetadata object or None if no video loaded
        """
        return self.metadata

    def get_playback_state(self) -> PlaybackState:
        """
        Get current playback state

        Returns:
            PlaybackState object
        """
        return self.playback_state

    def export_frame_sequence(
        self,
        output_dir: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        format: str = "jpg",
    ) -> bool:
        """
        Export a sequence of frames to files

        Args:
            output_dir: Output directory
            start_frame: Starting frame number
            end_frame: Ending frame number
            format: Image format (jpg, png, bmp)

        Returns:
            bool: True if export successful
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            frames = self.extract_frames(start_frame, end_frame)

            for i, frame in enumerate(frames):
                frame_num = start_frame + i
                filename = f"frame_{frame_num:06d}.{format}"
                file_path = output_path / filename

                cv2.imwrite(str(file_path), frame)

            logger.info(f"Exported {len(frames)} frames to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to export frame sequence: {e}")
            return False

    def close_video(self) -> None:
        """Close the current video and clean up resources"""
        # Stop playback
        self.stop()

        # Wait for playback thread to finish
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)

        # Clean up OpenCV capture
        if self.cap:
            self.cap.release()
            self.cap = None

        # Clean up MoviePy clip
        if self.video_clip:
            self.video_clip.close()
            self.video_clip = None

        # Reset state
        self.current_video_path = None
        self.metadata = None
        self.playback_state = PlaybackState()
        self.frame_buffer.clear()

        logger.info("Video closed and resources cleaned up")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_video()

    def _extract_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from video file"""
        try:
            # Basic metadata from OpenCV
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            # File size
            size_bytes = video_path.stat().st_size

            # Try to get additional metadata using ffmpeg
            codec = "unknown"
            bitrate = None

            if HAS_FFMPEG:
                try:
                    probe = ffmpeg.probe(str(video_path))
                    video_stream = next(
                        s for s in probe["streams"] if s["codec_type"] == "video"
                    )
                    codec = video_stream.get("codec_name", "unknown")
                    bitrate = (
                        int(video_stream.get("bit_rate", 0))
                        if video_stream.get("bit_rate")
                        else None
                    )
                except Exception as e:
                    logger.warning(f"Could not extract extended metadata: {e}")
            else:
                logger.debug("FFmpeg not available for extended metadata")

            return VideoMetadata(
                filename=video_path.name,
                duration=duration,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                format=video_path.suffix.lower(),
                size_bytes=size_bytes,
                codec=codec,
                bitrate=bitrate,
            )

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            raise VideoProcessingError(f"Failed to extract metadata: {e}")

    def _playback_loop(self) -> None:
        """Main playback loop running in separate thread"""
        try:
            while not self._stop_playback.is_set():
                with self.playback_lock:
                    if (
                        not self.playback_state.is_playing
                        or self.playback_state.is_paused
                    ):
                        time.sleep(0.01)
                        continue

                # Calculate frame timing
                target_fps = self.metadata.fps / self.playback_state.playback_speed
                frame_delay = 1.0 / target_fps if target_fps > 0 else 0.033

                start_time = time.time()

                # Get current frame
                frame = self.get_current_frame()
                if frame is None:
                    # End of video reached
                    self.stop()
                    break

                # Update playback state
                with self.playback_lock:
                    self.playback_state.current_frame += 1
                    self.playback_state.current_time = (
                        self.playback_state.current_frame / self.metadata.fps
                    )

                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame, self.playback_state.current_frame)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")

                # Timing control
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Playback loop error: {e}")
            self.stop()

    def _load_frame_buffer(self, start_frame: int, count: int) -> None:
        """Pre-load frames into buffer for smooth playback"""
        try:
            self.frame_buffer.clear()
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

            for i in range(count):
                frame_num = start_frame + i
                if frame_num >= self.metadata.frame_count:
                    break

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.cap.read()
                if ret:
                    self.frame_buffer.append(frame)

            # Restore position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            logger.info(f"Loaded {len(self.frame_buffer)} frames into buffer")

        except Exception as e:
            logger.error(f"Failed to load frame buffer: {e}")

    def _log_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log events and call event callbacks"""
        if self.enable_logging:
            event_data["timestamp"] = time.time()
            event_data["video_path"] = self.current_video_path

            # Call event callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event_type, event_data)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")


# Integration helper functions for CCTV analysis


def create_cctv_player(**kwargs) -> VideoPlayer:
    """
    Create a video player optimized for CCTV analysis

    Returns:
        VideoPlayer configured for CCTV use
    """
    player = VideoPlayer(**kwargs)

    # Add CCTV-specific frame callback for motion detection integration
    def cctv_frame_callback(frame: np.ndarray, frame_number: int):
        # Placeholder for YOLO integration
        # This would integrate with src/detection/yolo_movement.py
        pass

    player.add_frame_callback(cctv_frame_callback)

    return player


def integrate_with_voice_commands(player: VideoPlayer, command_parser) -> None:
    """
    Integrate video player with voice command system

    Args:
        player: VideoPlayer instance
        command_parser: CommandParser from src/nlp/parse_command.py
    """

    def handle_voice_command(command: str) -> bool:
        """Handle voice commands for video control"""
        try:
            parsed = command_parser.parse_command(command.lower())
            actions = [action.lower() for action in parsed.get("actions", [])]

            if "play" in actions or "start" in actions:
                return player.play()
            elif "pause" in actions:
                return player.pause()
            elif "stop" in actions:
                return player.stop()
            elif "seek" in actions or "jump" in actions:
                # Extract time from command (simplified)
                objects = parsed.get("objects", [])
                for obj in objects:
                    if obj.isdigit():
                        time_seconds = int(obj)
                        return player.seek_to_time(time_seconds)
            elif "speed" in actions:
                # Extract speed multiplier
                objects = parsed.get("objects", [])
                for obj in objects:
                    try:
                        speed = float(obj)
                        return player.set_playback_speed(speed)
                    except ValueError:
                        continue

            return False

        except Exception as e:
            logger.error(f"Voice command handling error: {e}")
            return False

    # Store the handler for external use
    player.voice_command_handler = handle_voice_command
    logger.info("Voice command integration completed")


# Example usage and testing functions


def demo_video_player():
    """Demonstration of video player capabilities"""
    player = VideoPlayer(display_size=(1024, 768), enable_audio=True)

    # Example callbacks
    def frame_processor(frame, frame_num):
        # Apply some processing
        player.apply_frame_filter(frame, "enhance", contrast=1.1)
        print(f"Processed frame {frame_num}")

    def event_logger(event_type, event_data):
        print(f"Event: {event_type} - {event_data}")

    player.add_frame_callback(frame_processor)
    player.add_event_callback(event_logger)

    return player


if __name__ == "__main__":
    # Simple test
    print("Video Player Module loaded successfully")
    print("Available classes: VideoPlayer, VideoMetadata, PlaybackState")
    print("Available exceptions: VideoProcessingError")

    # Create a demo player
    demo_player = demo_video_player()
    print(f"Demo player created with display size: {demo_player.display_size}")
    demo_player.close_video()
