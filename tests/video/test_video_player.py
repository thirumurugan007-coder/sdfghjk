"""
Test VideoPlayer functionality.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from video.video_player import VideoPlayer


@pytest.fixture
def sample_video():
    """Create a sample video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        video_path = tmp_file.name
        
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    # Create 90 frames (3 seconds at 30fps)
    for i in range(90):
        # Create a frame with changing colors
        frame = np.ones((480, 640, 3), dtype=np.uint8) * (i % 255)
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup
    if os.path.exists(video_path):
        os.unlink(video_path)


class TestVideoPlayer:
    """Test cases for VideoPlayer class."""
    
    def test_initialization(self):
        """Test VideoPlayer initialization."""
        player = VideoPlayer()
        
        assert player.cap is None
        assert player.video_path is None
        assert not player.is_playing
        assert not player.is_paused
        assert player.current_frame == 0
        assert player.total_frames == 0
        assert player.fps == 30.0
        assert player.playback_speed == 1.0
        
    def test_load_video_success(self, sample_video):
        """Test successful video loading."""
        player = VideoPlayer()
        result = player.load_video(sample_video)
        
        assert result is True
        assert player.video_path == sample_video
        assert player.total_frames == 90
        assert player.fps == 30.0
        assert player.video_width == 640
        assert player.video_height == 480
        
    def test_load_video_file_not_found(self):
        """Test loading non-existent video file."""
        player = VideoPlayer()
        
        with pytest.raises(FileNotFoundError):
            player.load_video("non_existent_video.mp4")
            
    def test_seek_frame(self, sample_video):
        """Test frame seeking functionality."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        # Test normal seek
        assert player.seek_frame(45) is True
        assert player.current_frame == 45
        
        # Test boundary conditions
        assert player.seek_frame(-1) is True
        assert player.current_frame == 0
        
        assert player.seek_frame(100) is True
        assert player.current_frame == 89  # Last frame
        
    def test_seek_time(self, sample_video):
        """Test time-based seeking."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        # Seek to 1.5 seconds (should be frame 45)
        assert player.seek_time(1.5) is True
        assert player.current_frame == 45
        
    def test_get_current_frame(self, sample_video):
        """Test getting current frame."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        frame = player.get_current_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        
    def test_get_frame_at(self, sample_video):
        """Test getting frame at specific position."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        # Get frame at position 30
        frame = player.get_frame_at(30)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        
        # Current position should be restored
        assert player.current_frame == 0
        
    def test_next_frame(self, sample_video):
        """Test advancing to next frame."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        initial_frame = player.current_frame
        frame = player.next_frame()
        
        assert frame is not None
        assert player.current_frame == initial_frame + 1
        
    def test_previous_frame(self, sample_video):
        """Test going to previous frame."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        # Move to frame 10 first
        player.seek_frame(10)
        
        frame = player.previous_frame()
        assert frame is not None
        assert player.current_frame == 9
        
    def test_get_video_info(self, sample_video):
        """Test getting video information."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        info = player.get_video_info()
        
        assert info["path"] == sample_video
        assert info["width"] == 640
        assert info["height"] == 480
        assert info["fps"] == 30.0
        assert info["total_frames"] == 90
        assert info["duration"] == 3.0
        
    def test_playback_speed(self, sample_video):
        """Test playback speed control."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        player.set_playback_speed(2.0)
        assert player.playback_speed == 2.0
        
        # Test minimum speed constraint
        player.set_playback_speed(0.05)
        assert player.playback_speed == 0.1
        
    def test_extract_frames(self, sample_video):
        """Test frame extraction."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        frames = player.extract_frames(0, 10, 2)
        
        assert len(frames) == 5  # Frames 0, 2, 4, 6, 8
        for frame in frames:
            assert frame.shape == (480, 640, 3)
            
    def test_get_progress_percentage(self, sample_video):
        """Test progress percentage calculation."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        # At beginning
        assert player.get_progress_percentage() == 0.0
        
        # At middle
        player.seek_frame(45)
        assert abs(player.get_progress_percentage() - 50.0) < 1.0
        
    def test_get_current_time(self, sample_video):
        """Test current time calculation."""
        player = VideoPlayer()
        player.load_video(sample_video)
        
        player.seek_frame(30)  # 1 second at 30fps
        assert abs(player.get_current_time() - 1.0) < 0.1