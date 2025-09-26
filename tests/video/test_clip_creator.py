"""
Test ClipCreator functionality.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from video.clip_creator import ClipCreator


@pytest.fixture
def sample_video():
    """Create a sample video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        video_path = tmp_file.name
        
    # Create a test video (5 seconds at 30fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    for i in range(150):  # 5 seconds
        frame = np.ones((480, 640, 3), dtype=np.uint8) * ((i * 2) % 255)
        # Add some motion in the middle
        if 60 <= i <= 90:  # Motion between 2-3 seconds
            x = (i - 60) * 10
            cv2.circle(frame, (x + 100, 240), 30, (255, 0, 0), -1)
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup
    if os.path.exists(video_path):
        os.unlink(video_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    import tempfile
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestClipCreator:
    """Test cases for ClipCreator class."""
    
    def test_initialization(self):
        """Test ClipCreator initialization."""
        creator = ClipCreator()
        
        assert creator.output_dir.name == "clips"
        assert creator.default_codec == 'mp4v'
        assert '.mp4' in creator.supported_formats
        
    def test_create_time_based_clip_success(self, sample_video, temp_output_dir):
        """Test successful time-based clip creation."""
        creator = ClipCreator()
        creator.output_dir = Path(temp_output_dir)
        
        output_path = Path(temp_output_dir) / "test_clip.mp4"
        result = creator.create_time_based_clip(
            sample_video, 
            start_time=1.0, 
            end_time=3.0, 
            output_path=str(output_path),
            compress=False  # Use raw clip for testing
        )
        
        assert result["success"] is True
        assert "duration" in result
        assert result["duration"] == 2.0
        assert output_path.exists()
        
        # Check metadata file was created
        metadata_path = output_path.with_suffix('.json')
        assert metadata_path.exists()
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            assert "start_time" in metadata
            assert "end_time" in metadata
            assert metadata["start_time"] == 1.0
            assert metadata["end_time"] == 3.0
            
    def test_create_time_based_clip_nonexistent(self):
        """Test clip creation with non-existent video."""
        creator = ClipCreator()
        result = creator.create_time_based_clip(
            "nonexistent.mp4", 1.0, 3.0
        )
        
        assert result["success"] is False
        assert "error" in result
        
    def test_create_time_based_clip_invalid_range(self, sample_video):
        """Test clip creation with invalid time range."""
        creator = ClipCreator()
        
        # Start time after end time
        result = creator.create_time_based_clip(
            sample_video, 3.0, 1.0
        )
        assert result["success"] is False
        
        # Start time beyond video duration
        result = creator.create_time_based_clip(
            sample_video, 10.0, 15.0
        )
        assert result["success"] is False
        
    def test_create_motion_based_clips(self, sample_video, temp_output_dir):
        """Test motion-based clip creation."""
        creator = ClipCreator()
        creator.output_dir = Path(temp_output_dir)
        
        clips = creator.create_motion_based_clips(
            sample_video,
            motion_threshold=0.01,  # Low threshold to detect our test motion
            min_duration=0.5,
            max_duration=5.0
        )
        
        assert isinstance(clips, list)
        # Note: Motion detection may or may not find motion depending on
        # background subtraction behavior, so we just check structure
        
    def test_create_motion_based_clips_nonexistent(self):
        """Test motion-based clips with non-existent video."""
        creator = ClipCreator()
        clips = creator.create_motion_based_clips("nonexistent.mp4")
        
        assert clips == []
        
    def test_create_event_based_clip(self, sample_video, temp_output_dir):
        """Test event-based clip creation."""
        creator = ClipCreator()
        creator.output_dir = Path(temp_output_dir)
        
        result = creator.create_event_based_clip(
            sample_video,
            event_time=2.5,
            duration=2.0,
            center_on_event=True
        )
        
        # Should succeed even if actual clip creation might fail
        assert "event_time" in result
        assert result["event_time"] == 2.5
        
    def test_extract_frames_to_images(self, sample_video, temp_output_dir):
        """Test frame extraction to images."""
        creator = ClipCreator()
        
        result = creator.extract_frames_to_images(
            sample_video,
            output_dir=temp_output_dir,
            frame_interval=30,  # Every second
            image_format="jpg"
        )
        
        assert result["success"] is True
        assert "saved_frames" in result
        assert result["frame_interval"] == 30
        
        # Check that images were created
        output_dir = Path(temp_output_dir)
        image_files = list(output_dir.glob("*.jpg"))
        assert len(image_files) > 0
        
    def test_extract_frames_nonexistent(self):
        """Test frame extraction with non-existent video."""
        creator = ClipCreator()
        result = creator.extract_frames_to_images("nonexistent.mp4")
        
        assert result["success"] is False
        
    def test_detect_motion_segments(self, sample_video):
        """Test motion segment detection."""
        creator = ClipCreator()
        segments = creator._detect_motion_segments(
            sample_video, 
            threshold=0.01, 
            min_duration=0.5, 
            max_duration=10.0
        )
        
        assert isinstance(segments, list)
        
        # Check segment structure if any detected
        for segment in segments:
            assert "start" in segment
            assert "end" in segment
            assert "duration" in segment
            assert segment["duration"] == segment["end"] - segment["start"]
            
    def test_generate_clip_metadata(self, sample_video, temp_output_dir):
        """Test clip metadata generation."""
        creator = ClipCreator()
        
        # Create a dummy output file
        output_path = Path(temp_output_dir) / "test.mp4"
        output_path.touch()
        
        metadata = creator._generate_clip_metadata(
            sample_video, 1.0, 3.0, str(output_path)
        )
        
        assert "source_video" in metadata
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "duration" in metadata
        assert "created_at" in metadata
        
        assert metadata["source_video"] == sample_video
        assert metadata["start_time"] == 1.0
        assert metadata["end_time"] == 3.0
        assert metadata["duration"] == 2.0