"""
Summary Generator Module for creating video summaries and reports.

This module provides functionality to analyze video content and generate
comprehensive summaries including activity timelines, motion reports,
and key event detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from io import BytesIO


@dataclass
class ActivityEvent:
    """Data class for activity events."""

    timestamp: float
    event_type: str
    confidence: float
    location: Tuple[int, int]
    description: str
    metadata: Dict[str, Any]


@dataclass
class VideoSummary:
    """Data class for video summary."""

    video_path: str
    duration: float
    total_frames: int
    fps: float
    analysis_timestamp: str
    activity_level: str
    motion_percentage: float
    key_events: List[ActivityEvent]
    quality_metrics: Dict[str, Any]
    timeline_data: Dict[str, Any]


class SummaryGenerator:
    """
    Generate comprehensive video summaries and analysis reports.

    Features:
    - Activity timeline generation
    - Motion analysis and reporting
    - Key event detection and summarization
    - Visual summary creation
    - Statistical analysis
    - Export to multiple formats
    """

    def __init__(self):
        self.motion_threshold = 0.05
        self.activity_threshold = 0.3
        self.min_event_duration = 1.0
        self.summary_interval = 10.0  # seconds

    def generate_comprehensive_summary(
        self, video_path: str, output_dir: Optional[str] = None
    ) -> VideoSummary:
        """
        Generate a comprehensive summary of the video.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save summary files

        Returns:
            VideoSummary object containing all analysis results
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_dir is None:
            output_dir = Path("./summaries")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Analyze video content
            analysis_results = self._analyze_video_content(cap, fps, duration)

            # Generate timeline
            timeline_data = self._generate_timeline(analysis_results, duration)

            # Detect key events
            key_events = self._detect_key_events(analysis_results, fps)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(analysis_results)

            # Determine overall activity level
            activity_level = self._determine_activity_level(analysis_results)

            # Calculate motion percentage
            motion_percentage = self._calculate_motion_percentage(analysis_results)

            # Create summary object
            summary = VideoSummary(
                video_path=video_path,
                duration=duration,
                total_frames=total_frames,
                fps=fps,
                analysis_timestamp=datetime.now().isoformat(),
                activity_level=activity_level,
                motion_percentage=motion_percentage,
                key_events=key_events,
                quality_metrics=quality_metrics,
                timeline_data=timeline_data,
            )

            # Save summary files
            self._save_summary_files(summary, output_dir)

            return summary

        finally:
            cap.release()

    def generate_activity_timeline(
        self, video_path: str, interval: float = 10.0
    ) -> Dict[str, Any]:
        """
        Generate an activity timeline for the video.

        Args:
            video_path: Path to the video file
            interval: Time interval for timeline segments in seconds

        Returns:
            Dictionary containing timeline data
        """
        if not Path(video_path).exists():
            return {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Calculate segments
            num_segments = int(np.ceil(duration / interval))
            timeline_segments = []

            bg_subtractor = cv2.createBackgroundSubtractorMOG2()

            for segment_idx in range(num_segments):
                segment_start = segment_idx * interval
                segment_end = min((segment_idx + 1) * interval, duration)

                # Analyze segment
                segment_data = self._analyze_segment(
                    cap, bg_subtractor, segment_start, segment_end, fps
                )

                timeline_segments.append(
                    {
                        "start_time": segment_start,
                        "end_time": segment_end,
                        "duration": segment_end - segment_start,
                        **segment_data,
                    }
                )

            return {
                "video_path": video_path,
                "total_duration": duration,
                "segment_interval": interval,
                "segments": timeline_segments,
                "generated_at": datetime.now().isoformat(),
            }

        finally:
            cap.release()

    def generate_motion_report(self, video_path: str) -> Dict[str, Any]:
        """
        Generate a detailed motion analysis report.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing motion analysis results
        """
        if not Path(video_path).exists():
            return {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            bg_subtractor = cv2.createBackgroundSubtractorMOG2()

            motion_data = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect motion
                fg_mask = bg_subtractor.apply(frame)
                motion_pixels = cv2.countNonZero(fg_mask)
                total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
                motion_percentage = (motion_pixels / total_pixels) * 100

                # Find motion areas
                contours, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                motion_areas = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Filter small areas
                        x, y, w, h = cv2.boundingRect(contour)
                        motion_areas.append({"bbox": [x, y, w, h], "area": area})

                motion_data.append(
                    {
                        "frame": frame_count,
                        "timestamp": frame_count / fps if fps > 0 else frame_count,
                        "motion_percentage": motion_percentage,
                        "motion_areas": motion_areas,
                    }
                )

                frame_count += 1

            # Calculate statistics
            motion_percentages = [d["motion_percentage"] for d in motion_data]

            return {
                "video_path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "motion_statistics": {
                    "mean_motion": np.mean(motion_percentages),
                    "max_motion": np.max(motion_percentages),
                    "min_motion": np.min(motion_percentages),
                    "std_motion": np.std(motion_percentages),
                    "frames_with_motion": len(
                        [p for p in motion_percentages if p > self.motion_threshold]
                    ),
                    "motion_percentage": (
                        len(
                            [p for p in motion_percentages if p > self.motion_threshold]
                        )
                        / len(motion_percentages)
                    )
                    * 100,
                },
                "motion_data": motion_data,
                "generated_at": datetime.now().isoformat(),
            }

        finally:
            cap.release()

    def create_visual_summary(
        self, summary: VideoSummary, output_path: Optional[str] = None
    ) -> str:
        """
        Create a visual summary chart of the video analysis.

        Args:
            summary: VideoSummary object
            output_path: Path to save the visual summary

        Returns:
            Path to the saved visual summary
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"visual_summary_{timestamp}.png"

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Video Analysis Summary: {Path(summary.video_path).name}",
            fontsize=16,
            fontweight="bold",
        )

        # Activity Timeline
        if summary.timeline_data and "segments" in summary.timeline_data:
            segments = summary.timeline_data["segments"]
            times = [s["start_time"] for s in segments]
            activities = [s.get("activity_score", 0) for s in segments]

            axes[0, 0].plot(times, activities, "b-", linewidth=2)
            axes[0, 0].fill_between(times, activities, alpha=0.3)
            axes[0, 0].set_title("Activity Timeline")
            axes[0, 0].set_xlabel("Time (seconds)")
            axes[0, 0].set_ylabel("Activity Level")
            axes[0, 0].grid(True, alpha=0.3)

        # Motion Distribution
        if summary.timeline_data and "segments" in summary.timeline_data:
            segments = summary.timeline_data["segments"]
            motion_levels = [s.get("motion_percentage", 0) for s in segments]

            axes[0, 1].hist(motion_levels, bins=20, alpha=0.7, color="orange")
            axes[0, 1].set_title("Motion Distribution")
            axes[0, 1].set_xlabel("Motion Percentage")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].grid(True, alpha=0.3)

        # Key Events Timeline
        if summary.key_events:
            event_times = [event.timestamp for event in summary.key_events]
            event_types = [event.event_type for event in summary.key_events]

            # Create scatter plot with different colors for different event types
            unique_types = list(set(event_types))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))

            for i, event_type in enumerate(unique_types):
                event_indices = [
                    j for j, t in enumerate(event_types) if t == event_type
                ]
                event_times_type = [event_times[j] for j in event_indices]

                axes[1, 0].scatter(
                    event_times_type,
                    [i] * len(event_times_type),
                    c=[colors[i]],
                    label=event_type,
                    s=50,
                )

            axes[1, 0].set_title("Key Events Timeline")
            axes[1, 0].set_xlabel("Time (seconds)")
            axes[1, 0].set_ylabel("Event Type")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Quality Metrics Bar Chart
        if summary.quality_metrics:
            metrics = ["Brightness", "Contrast", "Sharpness", "Overall"]
            values = [
                summary.quality_metrics.get("brightness_score", 0),
                summary.quality_metrics.get("contrast_score", 0),
                summary.quality_metrics.get("sharpness_score", 0),
                summary.quality_metrics.get("overall_score", 0),
            ]

            # Normalize values to 0-1 range
            values = [min(1.0, max(0.0, v)) for v in values]

            bars = axes[1, 1].bar(
                metrics, values, color=["blue", "green", "orange", "red"], alpha=0.7
            )
            axes[1, 1].set_title("Quality Metrics")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def export_summary_report(
        self, summary: VideoSummary, format_type: str = "json"
    ) -> str:
        """
        Export summary report in specified format.

        Args:
            summary: VideoSummary object
            format_type: Export format ('json', 'html', 'txt')

        Returns:
            Path to the exported report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type == "json":
            output_path = f"summary_report_{timestamp}.json"
            with open(output_path, "w") as f:
                json.dump(asdict(summary), f, indent=2, default=str)

        elif format_type == "html":
            output_path = f"summary_report_{timestamp}.html"
            self._export_html_report(summary, output_path)

        elif format_type == "txt":
            output_path = f"summary_report_{timestamp}.txt"
            self._export_text_report(summary, output_path)

        return output_path

    def _analyze_video_content(
        self, cap: cv2.VideoCapture, fps: float, duration: float
    ) -> Dict[str, Any]:
        """Analyze video content frame by frame."""
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        frame_data = []
        frame_count = 0

        # Sample frames for analysis (every 10th frame for efficiency)
        frame_step = max(1, int(fps / 3))  # ~3 frames per second

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                timestamp = frame_count / fps if fps > 0 else frame_count

                # Analyze frame
                frame_analysis = self._analyze_single_frame(
                    frame, bg_subtractor, timestamp
                )
                frame_data.append(frame_analysis)

            frame_count += 1

        return {
            "frame_data": frame_data,
            "total_frames_analyzed": len(frame_data),
            "sample_rate": frame_step,
        }

    def _analyze_single_frame(
        self, frame: np.ndarray, bg_subtractor, timestamp: float
    ) -> Dict[str, Any]:
        """Analyze a single frame for various metrics."""
        # Motion detection
        fg_mask = bg_subtractor.apply(frame)
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100

        # Brightness analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Activity score (combination of motion and edge density)
        edges = cv2.Canny(frame, 50, 150)
        edge_density = np.count_nonzero(edges) / total_pixels
        activity_score = (motion_percentage / 100 * 0.7) + (edge_density * 0.3)

        return {
            "timestamp": timestamp,
            "motion_percentage": motion_percentage,
            "brightness": brightness,
            "edge_density": edge_density,
            "activity_score": activity_score,
        }

    def _generate_timeline(
        self, analysis_results: Dict[str, Any], duration: float
    ) -> Dict[str, Any]:
        """Generate timeline data from analysis results."""
        frame_data = analysis_results.get("frame_data", [])

        # Group by intervals
        interval = self.summary_interval
        num_intervals = int(np.ceil(duration / interval))

        timeline_segments = []

        for i in range(num_intervals):
            start_time = i * interval
            end_time = min((i + 1) * interval, duration)

            # Find frames in this interval
            interval_frames = [
                f for f in frame_data if start_time <= f["timestamp"] < end_time
            ]

            if interval_frames:
                avg_motion = np.mean([f["motion_percentage"] for f in interval_frames])
                avg_brightness = np.mean([f["brightness"] for f in interval_frames])
                avg_activity = np.mean([f["activity_score"] for f in interval_frames])
                max_motion = np.max([f["motion_percentage"] for f in interval_frames])
            else:
                avg_motion = avg_brightness = avg_activity = max_motion = 0

            timeline_segments.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "motion_percentage": avg_motion,
                    "brightness": avg_brightness,
                    "activity_score": avg_activity,
                    "max_motion": max_motion,
                    "frame_count": len(interval_frames),
                }
            )

        return {"segments": timeline_segments, "interval": interval}

    def _detect_key_events(
        self, analysis_results: Dict[str, Any], fps: float
    ) -> List[ActivityEvent]:
        """Detect key events from analysis results."""
        frame_data = analysis_results.get("frame_data", [])
        events = []

        # Detect high activity events
        for frame in frame_data:
            if frame["activity_score"] > self.activity_threshold:
                event = ActivityEvent(
                    timestamp=frame["timestamp"],
                    event_type="high_activity",
                    confidence=min(1.0, frame["activity_score"]),
                    location=(0, 0),  # Would need object detection for precise location
                    description=(
                        f"High activity detected "
                        f"(score: {frame['activity_score']:.2f})"
                    ),
                    metadata={
                        "motion_percentage": frame["motion_percentage"],
                        "brightness": frame["brightness"],
                    },
                )
                events.append(event)

        # Merge nearby events
        merged_events = self._merge_nearby_events(events)

        return merged_events

    def _merge_nearby_events(self, events: List[ActivityEvent]) -> List[ActivityEvent]:
        """Merge events that are close in time."""
        if not events:
            return []

        merged = []
        current_group = [events[0]]

        for event in events[1:]:
            # If event is within merge threshold, add to current group
            if event.timestamp - current_group[-1].timestamp <= self.min_event_duration:
                current_group.append(event)
            else:
                # Create merged event from current group
                if current_group:
                    merged_event = self._create_merged_event(current_group)
                    merged.append(merged_event)
                current_group = [event]

        # Don't forget the last group
        if current_group:
            merged_event = self._create_merged_event(current_group)
            merged.append(merged_event)

        return merged

    def _create_merged_event(self, event_group: List[ActivityEvent]) -> ActivityEvent:
        """Create a single event from a group of events."""
        start_time = event_group[0].timestamp
        end_time = event_group[-1].timestamp
        duration = end_time - start_time

        avg_confidence = np.mean([e.confidence for e in event_group])

        return ActivityEvent(
            timestamp=start_time,
            event_type=f"{event_group[0].event_type}_sequence",
            confidence=avg_confidence,
            location=event_group[0].location,
            description=(
                f"Activity sequence from {start_time:.1f}s to {end_time:.1f}s "
                f"(duration: {duration:.1f}s)"
            ),
            metadata={
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "event_count": len(event_group),
            },
        )

    def _calculate_quality_metrics(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall quality metrics."""
        frame_data = analysis_results.get("frame_data", [])

        if not frame_data:
            return {}

        brightness_values = [f["brightness"] for f in frame_data]

        return {
            "brightness_score": np.mean(brightness_values) / 255.0,
            "contrast_score": np.std(brightness_values) / 128.0,
            "sharpness_score": 0.8,  # Would need proper sharpness calculation
            "overall_score": 0.75,  # Combined score
            "frame_count": len(frame_data),
        }

    def _determine_activity_level(self, analysis_results: Dict[str, Any]) -> str:
        """Determine overall activity level."""
        frame_data = analysis_results.get("frame_data", [])

        if not frame_data:
            return "unknown"

        avg_activity = np.mean([f["activity_score"] for f in frame_data])

        if avg_activity > 0.6:
            return "high"
        elif avg_activity > 0.3:
            return "medium"
        else:
            return "low"

    def _calculate_motion_percentage(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall motion percentage."""
        frame_data = analysis_results.get("frame_data", [])

        if not frame_data:
            return 0.0

        motion_frames = len(
            [f for f in frame_data if f["motion_percentage"] > self.motion_threshold]
        )
        return (motion_frames / len(frame_data)) * 100.0

    def _analyze_segment(
        self,
        cap: cv2.VideoCapture,
        bg_subtractor,
        start_time: float,
        end_time: float,
        fps: float,
    ) -> Dict[str, Any]:
        """Analyze a specific time segment."""
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        segment_data = []

        for frame_num in range(
            start_frame, min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / fps if fps > 0 else frame_num
            frame_analysis = self._analyze_single_frame(frame, bg_subtractor, timestamp)
            segment_data.append(frame_analysis)

        if segment_data:
            return {
                "motion_percentage": np.mean(
                    [f["motion_percentage"] for f in segment_data]
                ),
                "brightness": np.mean([f["brightness"] for f in segment_data]),
                "activity_score": np.mean([f["activity_score"] for f in segment_data]),
                "max_motion": np.max([f["motion_percentage"] for f in segment_data]),
                "frame_count": len(segment_data),
            }
        else:
            return {
                "motion_percentage": 0,
                "brightness": 0,
                "activity_score": 0,
                "max_motion": 0,
                "frame_count": 0,
            }

    def _save_summary_files(self, summary: VideoSummary, output_dir: Path):
        """Save summary files to output directory."""
        # Save JSON summary
        json_path = output_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)

        # Create and save visual summary
        visual_path = output_dir / "visual_summary.png"
        self.create_visual_summary(summary, str(visual_path))

    def _export_html_report(self, summary: VideoSummary, output_path: str):
        """Export summary as HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .event {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Video Analysis Report</h1>
                <p><strong>Video:</strong> {summary.video_path}</p>
                <p><strong>Duration:</strong> {summary.duration:.2f} seconds</p>
                <p><strong>Analysis Date:</strong> {summary.analysis_timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Overall Summary</h2>
                <div class="metric"><strong>Activity Level:</strong> {summary.activity_level}</div>
                <div class="metric">
                    <strong>Motion Percentage:</strong> 
                    {summary.motion_percentage:.2f}%
                </div>
                <div class="metric"><strong>Total Frames:</strong> {summary.total_frames}</div>
            </div>
            
            <div class="section">
                <h2>Key Events ({len(summary.key_events)})</h2>
                {"".join([
                    f'<div class="event">'
                    f'<strong>{event.timestamp:.1f}s:</strong> '
                    f'{event.description}'
                    f'</div>'
                    for event in summary.key_events
                ])}
            </div>
            
            <div class="section">
                <h2>Quality Metrics</h2>
                <div class="metric">
                    <strong>Overall Score:</strong>
                    {summary.quality_metrics.get('overall_score', 0):.2f}
                </div>
                <div class="metric">
                    <strong>Brightness Score:</strong>
                    {summary.quality_metrics.get('brightness_score', 0):.2f}
                </div>
                <div class="metric">
                    <strong>Contrast Score:</strong>
                    {summary.quality_metrics.get('contrast_score', 0):.2f}
                </div>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

    def _export_text_report(self, summary: VideoSummary, output_path: str):
        """Export summary as text report."""
        text_content = f"""
VIDEO ANALYSIS REPORT
=====================

Video File: {summary.video_path}
Duration: {summary.duration:.2f} seconds
Total Frames: {summary.total_frames}
FPS: {summary.fps:.2f}
Analysis Date: {summary.analysis_timestamp}

OVERALL SUMMARY
===============
Activity Level: {summary.activity_level}
Motion Percentage: {summary.motion_percentage:.2f}%

KEY EVENTS ({len(summary.key_events)})
==========
"""

        for event in summary.key_events:
            text_content += f"{event.timestamp:.1f}s: {event.description}\n"

        text_content += f"""

QUALITY METRICS
===============
Overall Score: {summary.quality_metrics.get('overall_score', 0):.2f}
Brightness Score: {summary.quality_metrics.get('brightness_score', 0):.2f}
Contrast Score: {summary.quality_metrics.get('contrast_score', 0):.2f}
Sharpness Score: {summary.quality_metrics.get('sharpness_score', 0):.2f}
"""

        with open(output_path, "w") as f:
            f.write(text_content)
