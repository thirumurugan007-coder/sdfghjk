"""
Video Processor Module for advanced video processing operations.

This module provides functionality for video analysis, frame processing,
motion detection, and various video transformations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime


class VideoProcessor:
    """
    Advanced video processing capabilities for CCTV analysis.

    Features:
    - Frame analysis and processing
    - Motion detection and tracking
    - Object detection integration
    - Video metadata analysis
    - Frame filtering and enhancement
    """

    def __init__(self):
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.frame_cache: Dict[int, np.ndarray] = {}
        self.motion_threshold: float = 0.05
        self.min_contour_area: int = 500

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive frame analysis.

        Args:
            frame: Input frame to analyze

        Returns:
            Dictionary containing analysis results
        """
        if frame is None or frame.size == 0:
            return {}

        # Basic frame properties
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1

        # Color analysis
        color_analysis = self._analyze_colors(frame)

        # Brightness and contrast analysis
        brightness_contrast = self._analyze_brightness_contrast(frame)

        # Motion analysis
        motion_info = self.detect_motion(frame)

        # Edge detection for activity analysis
        edges = cv2.Canny(frame, 50, 150)
        edge_density = np.count_nonzero(edges) / (width * height)

        return {
            "timestamp": datetime.now().isoformat(),
            "dimensions": {"width": width, "height": height, "channels": channels},
            "color_analysis": color_analysis,
            "brightness_contrast": brightness_contrast,
            "motion_info": motion_info,
            "edge_density": edge_density,
            "activity_level": self._calculate_activity_level(motion_info, edge_density),
        }

    def detect_motion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect motion in frame using background subtraction.

        Args:
            frame: Input frame for motion detection

        Returns:
            Dictionary containing motion information
        """
        if frame is None or frame.size == 0:
            return {"has_motion": False, "motion_percentage": 0.0, "motion_areas": []}

        # Apply background subtraction
        fg_mask = self.motion_detector.apply(frame)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Calculate motion percentage
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_percentage = (motion_pixels / total_pixels) * 100

        # Find motion contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter significant motion areas
        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append(
                    {
                        "bbox": [x, y, w, h],
                        "area": area,
                        "center": [x + w // 2, y + h // 2],
                    }
                )

        has_motion = motion_percentage > self.motion_threshold

        return {
            "has_motion": has_motion,
            "motion_percentage": motion_percentage,
            "motion_areas": motion_areas,
            "motion_mask": fg_mask,
        }

    def enhance_frame(
        self, frame: np.ndarray, enhancement_type: str = "auto"
    ) -> np.ndarray:
        """
        Apply various enhancement techniques to improve frame quality.

        Args:
            frame: Input frame to enhance
            enhancement_type: Type of enhancement ('auto', 'brightness',
                            'contrast', 'sharpen', 'denoise')

        Returns:
            Enhanced frame
        """
        if frame is None or frame.size == 0:
            return frame

        enhanced = frame.copy()

        if enhancement_type == "auto":
            # Automatic enhancement
            enhanced = self._auto_enhance(enhanced)
        elif enhancement_type == "brightness":
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=30)
        elif enhancement_type == "contrast":
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)
        elif enhancement_type == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        elif enhancement_type == "denoise":
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced)

        return enhanced

    def extract_keyframes(
        self, video_path: str, threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Extract keyframes from video based on scene changes.

        Args:
            video_path: Path to video file
            threshold: Threshold for scene change detection

        Returns:
            List of keyframe information
        """
        keyframes = []

        if not Path(video_path).exists():
            return keyframes

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return keyframes

        try:
            prev_frame = None
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = self._calculate_frame_difference(prev_frame, frame)

                    # If difference exceeds threshold, it's a keyframe
                    if diff > threshold:
                        timestamp = frame_count / fps if fps > 0 else frame_count
                        keyframes.append(
                            {
                                "frame_number": frame_count,
                                "timestamp": timestamp,
                                "difference_score": diff,
                                "frame": frame.copy(),
                            }
                        )

                prev_frame = frame.copy()
                frame_count += 1

        finally:
            cap.release()

        return keyframes

    def create_video_thumbnail(
        self, video_path: str, timestamp: float = None
    ) -> Optional[np.ndarray]:
        """
        Create thumbnail from video at specified timestamp.

        Args:
            video_path: Path to video file
            timestamp: Time in seconds (None for middle of video)

        Returns:
            Thumbnail frame or None if error
        """
        if not Path(video_path).exists():
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if timestamp is None:
                # Use middle of video
                frame_number = total_frames // 2
            else:
                frame_number = int(timestamp * fps)

            frame_number = max(0, min(frame_number, total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                # Resize to thumbnail size
                thumbnail = cv2.resize(frame, (320, 240))
                return thumbnail

        finally:
            cap.release()

        return None

    def analyze_video_quality(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze overall video quality metrics.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary containing quality metrics
        """
        quality_metrics = {
            "overall_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "sharpness_score": 0.0,
            "noise_level": 0.0,
            "motion_smoothness": 0.0,
        }

        if not Path(video_path).exists():
            return quality_metrics

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return quality_metrics

        try:
            frame_count = 0
            brightness_scores = []
            contrast_scores = []
            sharpness_scores = []
            noise_scores = []

            # Sample frames for analysis
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_step = max(1, total_frames // 50)  # Sample ~50 frames

            for i in range(0, total_frames, sample_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Analyze frame quality
                frame_quality = self._analyze_frame_quality(frame)
                brightness_scores.append(frame_quality["brightness"])
                contrast_scores.append(frame_quality["contrast"])
                sharpness_scores.append(frame_quality["sharpness"])
                noise_scores.append(frame_quality["noise"])

                frame_count += 1

            if frame_count > 0:
                quality_metrics["brightness_score"] = np.mean(brightness_scores)
                quality_metrics["contrast_score"] = np.mean(contrast_scores)
                quality_metrics["sharpness_score"] = np.mean(sharpness_scores)
                quality_metrics["noise_level"] = np.mean(noise_scores)

                # Calculate overall score
                overall = (
                    quality_metrics["brightness_score"] * 0.2
                    + quality_metrics["contrast_score"] * 0.3
                    + quality_metrics["sharpness_score"] * 0.4
                    + (1.0 - quality_metrics["noise_level"]) * 0.1
                )
                quality_metrics["overall_score"] = overall

        finally:
            cap.release()

        return quality_metrics

    def _analyze_colors(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in frame."""
        if len(frame.shape) != 3:
            return {"dominant_color": [0, 0, 0], "color_variance": 0.0}

        # Convert to RGB for analysis
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate mean color
        mean_color = np.mean(rgb_frame.reshape(-1, 3), axis=0)

        # Calculate color variance
        color_variance = np.var(rgb_frame.reshape(-1, 3))

        return {
            "dominant_color": mean_color.tolist(),
            "color_variance": float(color_variance),
        }

    def _analyze_brightness_contrast(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze brightness and contrast."""
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        )

        brightness = np.mean(gray)
        contrast = np.std(gray)

        return {"brightness": float(brightness), "contrast": float(contrast)}

    def _calculate_activity_level(
        self, motion_info: Dict[str, Any], edge_density: float
    ) -> str:
        """Calculate overall activity level in frame."""
        motion_score = motion_info.get("motion_percentage", 0) / 100.0
        edge_score = min(edge_density * 10, 1.0)  # Normalize edge density

        combined_score = (motion_score * 0.7) + (edge_score * 0.3)

        if combined_score > 0.6:
            return "high"
        elif combined_score > 0.3:
            return "medium"
        else:
            return "low"

    def _auto_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply automatic enhancement based on frame analysis."""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_l, lab_a, lab_b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_l = clahe.apply(lab_l)

        # Merge channels and convert back
        enhanced = cv2.merge([lab_l, lab_a, lab_b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _calculate_frame_difference(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        """Calculate normalized difference between two frames."""
        if frame1.shape != frame2.shape:
            return 1.0

        # Convert to grayscale for comparison
        gray1 = (
            cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            if len(frame1.shape) == 3
            else frame1
        )
        gray2 = (
            cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            if len(frame2.shape) == 3
            else frame2
        )

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Normalize difference
        return np.mean(diff) / 255.0

    def _analyze_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze quality metrics for a single frame."""
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        )

        # Brightness (0-1 scale)
        brightness = np.mean(gray) / 255.0

        # Contrast (0-1 scale)
        contrast = np.std(gray) / 128.0  # Normalize by half the range

        # Sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian) / 10000.0  # Normalize
        sharpness = min(sharpness, 1.0)

        # Noise estimation
        noise = self._estimate_noise(gray)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "noise": noise,
        }

    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in grayscale image."""
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        noise_estimate = np.var(laplacian) / 10000.0
        return min(noise_estimate, 1.0)
