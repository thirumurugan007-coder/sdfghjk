"""
Test SummaryGenerator functionality.
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

from video.summary_generator import SummaryGenerator, VideoSummary, ActivityEvent


@pytest.fixture
def sample_video():
    """Create a sample video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        video_path = tmp_file.name
        
    # Create a test video with varying activity
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (320, 240))  # Smaller for faster processing
    
    for i in range(90):  # 3 seconds
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add activity in middle section
        if 30 <= i <= 60:
            # Add moving elements
            x = (i - 30) * 5
            cv2.circle(frame, (x + 50, 120), 20, (255, 255, 255), -1)
            # Add some brightness variation
            frame[:, :] = frame[:, :] + (i % 50)
            
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


class TestSummaryGenerator:
    """Test cases for SummaryGenerator class."""
    
    def test_initialization(self):
        """Test SummaryGenerator initialization."""
        generator = SummaryGenerator()
        
        assert generator.motion_threshold == 0.05
        assert generator.activity_threshold == 0.3
        assert generator.min_event_duration == 1.0
        assert generator.summary_interval == 10.0
        
    def test_generate_comprehensive_summary(self, sample_video, temp_output_dir):
        """Test comprehensive summary generation."""
        generator = SummaryGenerator()
        
        try:
            summary = generator.generate_comprehensive_summary(
                sample_video, 
                output_dir=temp_output_dir
            )
            
            assert isinstance(summary, VideoSummary)
            assert summary.video_path == sample_video
            assert summary.duration == 3.0  # 90 frames at 30fps
            assert summary.total_frames == 90
            assert summary.fps == 30.0
            assert summary.activity_level in ["low", "medium", "high"]
            assert isinstance(summary.motion_percentage, float)
            assert isinstance(summary.key_events, list)
            assert isinstance(summary.quality_metrics, dict)
            assert isinstance(summary.timeline_data, dict)
            
            # Check that output files were created
            output_dir = Path(temp_output_dir)
            assert (output_dir / "summary.json").exists()
            assert (output_dir / "visual_summary.png").exists()
            
        except Exception as e:
            # If matplotlib is not available, that's acceptable for basic testing
            if "matplotlib" in str(e).lower():
                pytest.skip("Matplotlib not available for visual summary generation")
            else:
                raise
                
    def test_generate_comprehensive_summary_nonexistent(self):
        """Test summary generation with non-existent video."""
        generator = SummaryGenerator()
        
        with pytest.raises(FileNotFoundError):
            generator.generate_comprehensive_summary("nonexistent.mp4")
            
    def test_generate_activity_timeline(self, sample_video):
        """Test activity timeline generation."""
        generator = SummaryGenerator()
        timeline = generator.generate_activity_timeline(sample_video, interval=1.0)
        
        assert "video_path" in timeline
        assert "total_duration" in timeline
        assert "segments" in timeline
        assert timeline["video_path"] == sample_video
        assert timeline["total_duration"] == 3.0
        
        segments = timeline["segments"]
        assert len(segments) == 3  # 3 one-second segments
        
        for segment in segments:
            assert "start_time" in segment
            assert "end_time" in segment
            assert "duration" in segment
            assert segment["duration"] == segment["end_time"] - segment["start_time"]
            
    def test_generate_activity_timeline_nonexistent(self):
        """Test timeline generation with non-existent video."""
        generator = SummaryGenerator()
        timeline = generator.generate_activity_timeline("nonexistent.mp4")
        
        assert timeline == {}
        
    def test_generate_motion_report(self, sample_video):
        """Test motion report generation."""
        generator = SummaryGenerator()
        report = generator.generate_motion_report(sample_video)
        
        assert "video_path" in report
        assert "total_frames" in report
        assert "fps" in report
        assert "motion_statistics" in report
        assert "motion_data" in report
        
        motion_stats = report["motion_statistics"]
        assert "mean_motion" in motion_stats
        assert "max_motion" in motion_stats
        assert "min_motion" in motion_stats
        assert "std_motion" in motion_stats
        assert "frames_with_motion" in motion_stats
        assert "motion_percentage" in motion_stats
        
    def test_generate_motion_report_nonexistent(self):
        """Test motion report with non-existent video."""
        generator = SummaryGenerator()
        report = generator.generate_motion_report("nonexistent.mp4")
        
        assert report == {}
        
    def test_export_summary_report_json(self, temp_output_dir):
        """Test JSON export of summary report."""
        generator = SummaryGenerator()
        
        # Create a sample summary
        summary = VideoSummary(
            video_path="test.mp4",
            duration=10.0,
            total_frames=300,
            fps=30.0,
            analysis_timestamp="2023-01-01T12:00:00",
            activity_level="medium",
            motion_percentage=25.5,
            key_events=[],
            quality_metrics={"overall_score": 0.8},
            timeline_data={"segments": []}
        )
        
        original_dir = os.getcwd()
        try:
            os.chdir(temp_output_dir)
            output_path = generator.export_summary_report(summary, "json")
            
            assert Path(output_path).exists()
            
            # Verify JSON content
            with open(output_path, 'r') as f:
                data = json.load(f)
                assert data["video_path"] == "test.mp4"
                assert data["duration"] == 10.0
                assert data["activity_level"] == "medium"
                
        finally:
            os.chdir(original_dir)
            
    def test_export_summary_report_html(self, temp_output_dir):
        """Test HTML export of summary report."""
        generator = SummaryGenerator()
        
        # Create a sample summary
        summary = VideoSummary(
            video_path="test.mp4",
            duration=10.0,
            total_frames=300,
            fps=30.0,
            analysis_timestamp="2023-01-01T12:00:00",
            activity_level="high",
            motion_percentage=75.0,
            key_events=[
                ActivityEvent(
                    timestamp=5.0,
                    event_type="motion",
                    confidence=0.8,
                    location=(100, 200),
                    description="Motion detected",
                    metadata={}
                )
            ],
            quality_metrics={"overall_score": 0.9},
            timeline_data={"segments": []}
        )
        
        original_dir = os.getcwd()
        try:
            os.chdir(temp_output_dir)
            output_path = generator.export_summary_report(summary, "html")
            
            assert Path(output_path).exists()
            
            # Verify HTML content
            with open(output_path, 'r') as f:
                html_content = f.read()
                assert "Video Analysis Report" in html_content
                assert "test.mp4" in html_content
                assert "high" in html_content
                
        finally:
            os.chdir(original_dir)
            
    def test_export_summary_report_txt(self, temp_output_dir):
        """Test text export of summary report."""
        generator = SummaryGenerator()
        
        # Create a sample summary
        summary = VideoSummary(
            video_path="test.mp4",
            duration=10.0,
            total_frames=300,
            fps=30.0,
            analysis_timestamp="2023-01-01T12:00:00",
            activity_level="low",
            motion_percentage=5.0,
            key_events=[],
            quality_metrics={"overall_score": 0.6},
            timeline_data={"segments": []}
        )
        
        original_dir = os.getcwd()
        try:
            os.chdir(temp_output_dir)
            output_path = generator.export_summary_report(summary, "txt")
            
            assert Path(output_path).exists()
            
            # Verify text content
            with open(output_path, 'r') as f:
                text_content = f.read()
                assert "VIDEO ANALYSIS REPORT" in text_content
                assert "test.mp4" in text_content
                assert "low" in text_content
                
        finally:
            os.chdir(original_dir)
            
    def test_analyze_single_frame(self):
        """Test single frame analysis."""
        generator = SummaryGenerator()
        
        # Create test frame
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        
        result = generator._analyze_single_frame(frame, bg_subtractor, 1.5)
        
        assert "timestamp" in result
        assert "motion_percentage" in result
        assert "brightness" in result
        assert "edge_density" in result
        assert "activity_score" in result
        
        assert result["timestamp"] == 1.5
        assert isinstance(result["motion_percentage"], float)
        assert isinstance(result["brightness"], float)
        
    def test_detect_key_events(self):
        """Test key event detection."""
        generator = SummaryGenerator()
        
        # Create analysis results with high activity
        analysis_results = {
            "frame_data": [
                {"timestamp": 1.0, "activity_score": 0.1, "motion_percentage": 5, "brightness": 128},
                {"timestamp": 2.0, "activity_score": 0.8, "motion_percentage": 80, "brightness": 120},
                {"timestamp": 3.0, "activity_score": 0.9, "motion_percentage": 85, "brightness": 125},
                {"timestamp": 4.0, "activity_score": 0.2, "motion_percentage": 10, "brightness": 130},
            ]
        }
        
        events = generator._detect_key_events(analysis_results, 30.0)
        
        assert isinstance(events, list)
        # Should detect events for high activity frames
        
    def test_determine_activity_level(self):
        """Test activity level determination."""
        generator = SummaryGenerator()
        
        # High activity
        analysis_results = {"frame_data": [{"activity_score": 0.8} for _ in range(10)]}
        level = generator._determine_activity_level(analysis_results)
        assert level == "high"
        
        # Medium activity
        analysis_results = {"frame_data": [{"activity_score": 0.5} for _ in range(10)]}
        level = generator._determine_activity_level(analysis_results)
        assert level == "medium"
        
        # Low activity
        analysis_results = {"frame_data": [{"activity_score": 0.1} for _ in range(10)]}
        level = generator._determine_activity_level(analysis_results)
        assert level == "low"
        
        # No data
        analysis_results = {"frame_data": []}
        level = generator._determine_activity_level(analysis_results)
        assert level == "unknown"
        
    def test_calculate_motion_percentage(self):
        """Test motion percentage calculation."""
        generator = SummaryGenerator()
        generator.motion_threshold = 0.1
        
        # Mix of motion and non-motion frames
        analysis_results = {
            "frame_data": [
                {"motion_percentage": 0.2},  # Motion
                {"motion_percentage": 0.05}, # No motion
                {"motion_percentage": 0.15}, # Motion
                {"motion_percentage": 0.02}, # No motion
            ]
        }
        
        motion_pct = generator._calculate_motion_percentage(analysis_results)
        assert motion_pct == 50.0  # 2 out of 4 frames have motion
        
        # No data
        analysis_results = {"frame_data": []}
        motion_pct = generator._calculate_motion_percentage(analysis_results)
        assert motion_pct == 0.0