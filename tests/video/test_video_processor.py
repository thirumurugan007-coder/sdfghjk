"""
Test VideoProcessor functionality.
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

from video.video_processor import VideoProcessor


@pytest.fixture
def sample_video():
    """Create a sample video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        video_path = tmp_file.name
        
    # Create a simple test video with motion
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    # Create frames with moving objects
    for i in range(60):  # 2 seconds
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a moving rectangle
        x = (i * 10) % 500
        y = 200
        cv2.rectangle(frame, (x, y), (x + 50, y + 50), (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup
    if os.path.exists(video_path):
        os.unlink(video_path)


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


class TestVideoProcessor:
    """Test cases for VideoProcessor class."""
    
    def test_initialization(self):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor()
        
        assert processor.motion_detector is not None
        assert processor.motion_threshold == 0.05
        assert processor.min_contour_area == 500
        
    def test_analyze_frame(self, sample_frame):
        """Test frame analysis."""
        processor = VideoProcessor()
        result = processor.analyze_frame(sample_frame)
        
        assert "timestamp" in result
        assert "dimensions" in result
        assert "color_analysis" in result
        assert "brightness_contrast" in result
        assert "motion_info" in result
        assert "activity_level" in result
        
        dimensions = result["dimensions"]
        assert dimensions["width"] == 640
        assert dimensions["height"] == 480
        assert dimensions["channels"] == 3
        
    def test_analyze_empty_frame(self):
        """Test analysis with empty frame."""
        processor = VideoProcessor()
        result = processor.analyze_frame(None)
        
        assert result == {}
        
    def test_detect_motion(self, sample_frame):
        """Test motion detection."""
        processor = VideoProcessor()
        
        # Process first frame to initialize background
        processor.detect_motion(sample_frame)
        
        # Process second frame
        result = processor.detect_motion(sample_frame)
        
        assert "has_motion" in result
        assert "motion_percentage" in result
        assert "motion_areas" in result
        assert "motion_mask" in result
        
        assert isinstance(result["has_motion"], bool)
        assert isinstance(result["motion_percentage"], float)
        assert isinstance(result["motion_areas"], list)
        
    def test_enhance_frame_auto(self, sample_frame):
        """Test automatic frame enhancement."""
        processor = VideoProcessor()
        enhanced = processor.enhance_frame(sample_frame, "auto")
        
        assert enhanced.shape == sample_frame.shape
        assert enhanced.dtype == sample_frame.dtype
        
    def test_enhance_frame_brightness(self, sample_frame):
        """Test brightness enhancement."""
        processor = VideoProcessor()
        enhanced = processor.enhance_frame(sample_frame, "brightness")
        
        assert enhanced.shape == sample_frame.shape
        
    def test_enhance_frame_contrast(self, sample_frame):
        """Test contrast enhancement."""
        processor = VideoProcessor()
        enhanced = processor.enhance_frame(sample_frame, "contrast")
        
        assert enhanced.shape == sample_frame.shape
        
    def test_enhance_frame_sharpen(self, sample_frame):
        """Test sharpening enhancement."""
        processor = VideoProcessor()
        enhanced = processor.enhance_frame(sample_frame, "sharpen")
        
        assert enhanced.shape == sample_frame.shape
        
    def test_enhance_empty_frame(self):
        """Test enhancement with empty frame."""
        processor = VideoProcessor()
        result = processor.enhance_frame(None, "auto")
        
        assert result is None
        
    def test_extract_keyframes(self, sample_video):
        """Test keyframe extraction."""
        processor = VideoProcessor()
        keyframes = processor.extract_keyframes(sample_video, threshold=0.1)
        
        assert isinstance(keyframes, list)
        
        if keyframes:  # If keyframes were detected
            for keyframe in keyframes:
                assert "frame_number" in keyframe
                assert "timestamp" in keyframe
                assert "difference_score" in keyframe
                assert "frame" in keyframe
                
    def test_extract_keyframes_nonexistent(self):
        """Test keyframe extraction with non-existent file."""
        processor = VideoProcessor()
        keyframes = processor.extract_keyframes("nonexistent.mp4")
        
        assert keyframes == []
        
    def test_create_video_thumbnail(self, sample_video):
        """Test thumbnail creation."""
        processor = VideoProcessor()
        thumbnail = processor.create_video_thumbnail(sample_video)
        
        assert thumbnail is not None
        assert thumbnail.shape == (240, 320, 3)  # Thumbnail size
        
    def test_create_video_thumbnail_with_timestamp(self, sample_video):
        """Test thumbnail creation at specific timestamp."""
        processor = VideoProcessor()
        thumbnail = processor.create_video_thumbnail(sample_video, timestamp=1.0)
        
        assert thumbnail is not None
        assert thumbnail.shape == (240, 320, 3)
        
    def test_create_video_thumbnail_nonexistent(self):
        """Test thumbnail creation with non-existent file."""
        processor = VideoProcessor()
        thumbnail = processor.create_video_thumbnail("nonexistent.mp4")
        
        assert thumbnail is None
        
    def test_analyze_video_quality(self, sample_video):
        """Test video quality analysis."""
        processor = VideoProcessor()
        quality = processor.analyze_video_quality(sample_video)
        
        assert "overall_score" in quality
        assert "brightness_score" in quality
        assert "contrast_score" in quality
        assert "sharpness_score" in quality
        assert "noise_level" in quality
        
        # Check that scores are within reasonable ranges
        assert 0 <= quality["overall_score"] <= 1
        assert 0 <= quality["brightness_score"] <= 1
        
    def test_analyze_video_quality_nonexistent(self):
        """Test quality analysis with non-existent file."""
        processor = VideoProcessor()
        quality = processor.analyze_video_quality("nonexistent.mp4")
        
        # Should return default metrics
        assert quality["overall_score"] == 0.0
        
    def test_analyze_colors(self):
        """Test color analysis."""
        processor = VideoProcessor()
        
        # Create a frame with known colors
        frame = np.full((100, 100, 3), [128, 64, 192], dtype=np.uint8)
        result = processor._analyze_colors(frame)
        
        assert "dominant_color" in result
        assert "color_variance" in result
        assert len(result["dominant_color"]) == 3
        
    def test_analyze_brightness_contrast(self):
        """Test brightness and contrast analysis."""
        processor = VideoProcessor()
        
        # Create a frame with known brightness
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = processor._analyze_brightness_contrast(frame)
        
        assert "brightness" in result
        assert "contrast" in result
        assert isinstance(result["brightness"], float)
        assert isinstance(result["contrast"], float)
        
    def test_calculate_activity_level(self):
        """Test activity level calculation."""
        processor = VideoProcessor()
        
        # High motion, low edge density
        motion_info = {"motion_percentage": 80}
        activity = processor._calculate_activity_level(motion_info, 0.1)
        assert activity in ["low", "medium", "high"]
        
        # Low motion, low edge density
        motion_info = {"motion_percentage": 5}
        activity = processor._calculate_activity_level(motion_info, 0.05)
        assert activity in ["low", "medium", "high"]
        
    def test_calculate_frame_difference(self):
        """Test frame difference calculation."""
        processor = VideoProcessor()
        
        # Same frames
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        diff = processor._calculate_frame_difference(frame1, frame2)
        assert diff == 0.0
        
        # Completely different frames
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)
        diff = processor._calculate_frame_difference(frame1, frame2)
        assert diff == 1.0
        
        # Different shaped frames
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((50, 50, 3), dtype=np.uint8)
        diff = processor._calculate_frame_difference(frame1, frame2)
        assert diff == 1.0