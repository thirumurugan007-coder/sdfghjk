"""
Clip Creator Module for generating video clips from CCTV footage.

This module provides functionality to create clips based on various criteria
such as motion detection, time ranges, and specific events.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import json
import ffmpeg


class ClipCreator:
    """
    Create video clips from CCTV footage based on various criteria.

    Features:
    - Time-based clip creation
    - Motion-triggered clips
    - Event-based clip generation
    - Clip metadata management
    - Multiple output formats
    """

    def __init__(self):
        self.output_dir = Path("./clips")
        self.output_dir.mkdir(exist_ok=True)
        self.supported_formats = [".mp4", ".avi", ".mov", ".mkv"]
        self.default_codec = "mp4v"

    def create_time_based_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a clip based on time range.

        Args:
            video_path: Path to source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path (auto-generated if None)
            compress: Whether to compress the output

        Returns:
            Dictionary with clip information
        """
        if not Path(video_path).exists():
            return {"success": False, "error": "Source video not found"}

        # Validate time range
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Cannot open source video"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if start_time >= duration or end_time <= start_time:
            return {"success": False, "error": "Invalid time range"}

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"clip_{timestamp}.mp4"
        else:
            output_path = Path(output_path)

        try:
            # Use ffmpeg for efficient clip creation
            if compress:
                # Create compressed clip
                clip_info = self._create_compressed_clip(
                    video_path, start_time, end_time, str(output_path)
                )
            else:
                # Create uncompressed clip
                clip_info = self._create_raw_clip(
                    video_path, start_time, end_time, str(output_path)
                )

            if clip_info["success"]:
                # Generate metadata
                metadata = self._generate_clip_metadata(
                    video_path, start_time, end_time, str(output_path)
                )
                clip_info.update(metadata)

                # Save metadata file
                metadata_path = output_path.with_suffix(".json")
                with open(metadata_path, "w") as f:
                    json.dump(clip_info, f, indent=2, default=str)

            return clip_info

        except Exception as e:
            return {"success": False, "error": f"Clip creation failed: {str(e)}"}

    def create_motion_based_clips(
        self,
        video_path: str,
        motion_threshold: float = 0.05,
        min_duration: float = 2.0,
        max_duration: float = 30.0,
        padding: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Create clips based on motion detection.

        Args:
            video_path: Path to source video
            motion_threshold: Motion sensitivity threshold
            min_duration: Minimum clip duration in seconds
            max_duration: Maximum clip duration in seconds
            padding: Padding before/after motion in seconds

        Returns:
            List of created clips information
        """
        clips = []

        if not Path(video_path).exists():
            return clips

        # Detect motion segments
        motion_segments = self._detect_motion_segments(
            video_path, motion_threshold, min_duration, max_duration
        )

        # Create clips for each motion segment
        for i, segment in enumerate(motion_segments):
            start_time = max(0, segment["start"] - padding)
            end_time = segment["end"] + padding

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"motion_clip_{timestamp}_{i:03d}.mp4"

            clip_info = self.create_time_based_clip(
                video_path, start_time, end_time, str(output_path)
            )

            if clip_info.get("success", False):
                clip_info["motion_data"] = segment
                clips.append(clip_info)

        return clips

    def create_event_based_clip(
        self,
        video_path: str,
        event_time: float,
        duration: float = 10.0,
        center_on_event: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a clip around a specific event time.

        Args:
            video_path: Path to source video
            event_time: Time of the event in seconds
            duration: Total clip duration
            center_on_event: Whether to center the clip on the event

        Returns:
            Dictionary with clip information
        """
        if center_on_event:
            start_time = max(0, event_time - duration / 2)
            end_time = start_time + duration
        else:
            start_time = event_time
            end_time = event_time + duration

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"event_clip_{timestamp}.mp4"

        clip_info = self.create_time_based_clip(
            video_path, start_time, end_time, str(output_path)
        )

        if clip_info.get("success", False):
            clip_info["event_time"] = event_time
            clip_info["event_centered"] = center_on_event

        return clip_info

    def create_highlight_reel(
        self,
        clips: List[str],
        output_path: Optional[str] = None,
        transition_duration: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Create a highlight reel from multiple clips.

        Args:
            clips: List of clip file paths
            output_path: Output file path
            transition_duration: Duration of transitions between clips

        Returns:
            Dictionary with highlight reel information
        """
        if not clips:
            return {"success": False, "error": "No clips provided"}

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"highlight_reel_{timestamp}.mp4"
        else:
            output_path = Path(output_path)

        try:
            # Use ffmpeg to concatenate clips with transitions
            return self._create_highlight_reel_ffmpeg(clips, str(output_path))
        except Exception as e:
            return {
                "success": False,
                "error": f"Highlight reel creation failed: {str(e)}",
            }

    def extract_frames_to_images(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        frame_interval: int = 30,
        image_format: str = "jpg",
    ) -> Dict[str, Any]:
        """
        Extract frames from video and save as images.

        Args:
            video_path: Path to source video
            output_dir: Output directory for images
            frame_interval: Extract every N frames
            image_format: Output image format (jpg, png)

        Returns:
            Dictionary with extraction information
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_dir / f"frames_{timestamp}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        if not Path(video_path).exists():
            return {"success": False, "error": "Source video not found"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Cannot open source video"}

        try:
            frame_count = 0
            saved_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_path = output_dir / f"frame_{frame_count:06d}.{image_format}"

                    if image_format.lower() == "jpg":
                        cv2.imwrite(
                            str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                        )
                    else:
                        cv2.imwrite(str(frame_path), frame)

                    saved_frames += 1

                frame_count += 1

            cap.release()

            return {
                "success": True,
                "output_dir": str(output_dir),
                "total_frames": frame_count,
                "saved_frames": saved_frames,
                "frame_interval": frame_interval,
            }

        except Exception as e:
            cap.release()
            return {"success": False, "error": f"Frame extraction failed: {str(e)}"}

    def _create_compressed_clip(
        self, video_path: str, start_time: float, end_time: float, output_path: str
    ) -> Dict[str, Any]:
        """Create compressed clip using ffmpeg."""
        try:
            duration = end_time - start_time

            stream = ffmpeg.input(video_path, ss=start_time, t=duration)
            stream = ffmpeg.output(
                stream,
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=23,  # Good quality compression
                preset="medium",
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            return {"success": True, "compressed": True}
        except FileNotFoundError:
            # ffmpeg not available, fall back to raw clip creation
            return self._create_raw_clip(video_path, start_time, end_time, output_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_raw_clip(
        self, video_path: str, start_time: float, end_time: float, output_path: str
    ) -> Dict[str, Any]:
        """Create uncompressed clip using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Set start frame
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.default_codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)
                current_frame += 1

            cap.release()
            out.release()

            return {"success": True, "compressed": False}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_motion_segments(
        self,
        video_path: str,
        threshold: float,
        min_duration: float,
        max_duration: float,
    ) -> List[Dict[str, Any]]:
        """Detect continuous motion segments in video."""
        segments = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return segments

        # Background subtractor for motion detection
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        fps = cap.get(cv2.CAP_PROP_FPS)

        motion_start = None
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect motion
                fg_mask = bg_subtractor.apply(frame)
                motion_pixels = cv2.countNonZero(fg_mask)
                total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
                motion_percentage = motion_pixels / total_pixels

                current_time = frame_count / fps

                if motion_percentage > threshold:
                    if motion_start is None:
                        motion_start = current_time
                else:
                    if motion_start is not None:
                        motion_duration = current_time - motion_start

                        if motion_duration >= min_duration:
                            # Limit to max duration
                            if motion_duration > max_duration:
                                # Split into multiple segments
                                start = motion_start
                                while start < current_time:
                                    end = min(start + max_duration, current_time)
                                    segments.append(
                                        {
                                            "start": start,
                                            "end": end,
                                            "duration": end - start,
                                            "motion_level": "high",
                                        }
                                    )
                                    start = end
                            else:
                                segments.append(
                                    {
                                        "start": motion_start,
                                        "end": current_time,
                                        "duration": motion_duration,
                                        "motion_level": "high",
                                    }
                                )

                        motion_start = None

                frame_count += 1

        finally:
            cap.release()

        return segments

    def _generate_clip_metadata(
        self, source_path: str, start_time: float, end_time: float, output_path: str
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for created clip."""
        output_file = Path(output_path)

        metadata = {
            "source_video": source_path,
            "output_path": output_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "created_at": datetime.now().isoformat(),
            "file_size": output_file.stat().st_size if output_file.exists() else 0,
        }

        # Get clip properties
        if output_file.exists():
            cap = cv2.VideoCapture(str(output_file))
            if cap.isOpened():
                metadata.update(
                    {
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    }
                )
                cap.release()

        return metadata

    def _create_highlight_reel_ffmpeg(
        self, clips: List[str], output_path: str
    ) -> Dict[str, Any]:
        """Create highlight reel using ffmpeg concatenation."""
        try:
            # Create temporary file list for ffmpeg concat
            file_list_path = self.output_dir / "temp_file_list.txt"

            with open(file_list_path, "w") as f:
                for clip_path in clips:
                    if Path(clip_path).exists():
                        f.write(f"file '{Path(clip_path).absolute()}'\n")

            # Use ffmpeg concat demuxer
            stream = ffmpeg.input(str(file_list_path), format="concat", safe=0)
            stream = ffmpeg.output(stream, output_path, c="copy")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            # Cleanup
            file_list_path.unlink(missing_ok=True)

            return {
                "success": True,
                "output_path": output_path,
                "input_clips": len(clips),
                "created_at": datetime.now().isoformat(),
            }

        except FileNotFoundError:
            return {
                "success": False,
                "error": "ffmpeg not available for highlight reel creation",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
