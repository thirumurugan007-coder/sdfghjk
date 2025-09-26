"""
Demo script for the Video module of CCTV Video Analyzer.

This script demonstrates the capabilities of the video module including:
- Video player functionality
- Video processing and analysis
- Clip creation
- Summary generation
"""

import sys
import os
import cv2
import numpy as np
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from video import VideoPlayer, VideoProcessor, ClipCreator, SummaryGenerator


def create_demo_video(output_path: str, duration_seconds: int = 10) -> str:
    """Create a demo video for testing purposes."""
    print(f"Creating demo video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    frames = duration_seconds * 30  # 30 fps
    
    for i in range(frames):
        # Create frame with animated content
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(480):
            frame[y, :] = [50 + y//10, 100, 150]
            
        # Moving circle (simulates motion)
        circle_x = int(100 + (i % 400))  # Move across screen
        circle_y = int(240 + 50 * np.sin(i * 0.1))  # Sine wave movement
        cv2.circle(frame, (circle_x, circle_y), 30, (255, 255, 0), -1)
        
        # Add some noise/activity in the middle section
        if 90 <= i <= 210:  # 3-7 second mark
            # Random rectangles for activity
            for _ in range(5):
                x = np.random.randint(0, 580)
                y = np.random.randint(0, 420)
                cv2.rectangle(frame, (x, y), (x+60, y+60), 
                            (np.random.randint(0, 255), 
                             np.random.randint(0, 255), 
                             np.random.randint(0, 255)), -1)
        
        # Add timestamp text
        timestamp = f"Time: {i/30.0:.1f}s"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Demo video created with {frames} frames ({duration_seconds} seconds)")
    return output_path


def demo_video_player(video_path: str):
    """Demonstrate video player functionality."""
    print("\n=== VIDEO PLAYER DEMO ===")
    
    player = VideoPlayer()
    
    # Load video
    if not player.load_video(video_path):
        print("Failed to load video")
        return
        
    print(f"Video loaded successfully: {video_path}")
    
    # Get video information
    info = player.get_video_info()
    print(f"Video info: {info}")
    
    # Test seeking
    print(f"Seeking to frame 150...")
    player.seek_frame(150)
    print(f"Current frame: {player.current_frame}")
    
    # Test time-based seeking
    print(f"Seeking to 3.5 seconds...")
    player.seek_time(3.5)
    print(f"Current time: {player.get_current_time():.2f}s")
    
    # Extract some frames
    frames = player.extract_frames(0, 60, 30)  # Every second for 2 seconds
    print(f"Extracted {len(frames)} frames")
    
    # Test playback speed
    player.set_playback_speed(2.0)
    print(f"Playback speed set to: {player.playback_speed}x")
    
    print("Video player demo completed!")


def demo_video_processor(video_path: str):
    """Demonstrate video processor functionality."""
    print("\n=== VIDEO PROCESSOR DEMO ===")
    
    processor = VideoProcessor()
    
    # Load video and get a frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame from video")
        return
        
    # Analyze frame
    analysis = processor.analyze_frame(frame)
    print("Frame analysis results:")
    for key, value in analysis.items():
        if key != 'motion_info':  # Skip detailed motion info for brevity
            print(f"  {key}: {value}")
    
    # Detect motion
    motion_info = processor.detect_motion(frame)
    print(f"Motion detected: {motion_info['has_motion']}")
    print(f"Motion percentage: {motion_info['motion_percentage']:.2f}%")
    
    # Enhance frame
    enhanced_frame = processor.enhance_frame(frame, "auto")
    print(f"Frame enhanced (shape: {enhanced_frame.shape})")
    
    # Create thumbnail
    thumbnail = processor.create_video_thumbnail(video_path, timestamp=5.0)
    if thumbnail is not None:
        print(f"Thumbnail created (shape: {thumbnail.shape})")
    
    # Analyze video quality
    quality = processor.analyze_video_quality(video_path)
    print("Video quality metrics:")
    for key, value in quality.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("Video processor demo completed!")


def demo_clip_creator(video_path: str, output_dir: str):
    """Demonstrate clip creator functionality."""
    print("\n=== CLIP CREATOR DEMO ===")
    
    creator = ClipCreator()
    creator.output_dir = Path(output_dir)
    
    # Create time-based clip
    clip_result = creator.create_time_based_clip(
        video_path,
        start_time=2.0,
        end_time=6.0,
        output_path=os.path.join(output_dir, "demo_clip.mp4"),
        compress=False  # Skip compression to avoid ffmpeg dependency
    )
    
    if clip_result.get("success"):
        print(f"Time-based clip created successfully!")
        print(f"  Duration: {clip_result['duration']}s")
        print(f"  Output: {clip_result.get('output_path', 'N/A')}")
    else:
        print(f"Clip creation failed: {clip_result.get('error', 'Unknown error')}")
    
    # Extract frames to images
    frames_result = creator.extract_frames_to_images(
        video_path,
        output_dir=os.path.join(output_dir, "frames"),
        frame_interval=60,  # Every 2 seconds
        image_format="jpg"
    )
    
    if frames_result.get("success"):
        print(f"Frame extraction completed!")
        print(f"  Total frames: {frames_result['total_frames']}")
        print(f"  Saved frames: {frames_result['saved_frames']}")
    else:
        print(f"Frame extraction failed: {frames_result.get('error', 'Unknown error')}")
    
    # Try motion-based clips (may not detect motion in our simple demo)
    motion_clips = creator.create_motion_based_clips(
        video_path,
        motion_threshold=0.01,  # Low threshold
        min_duration=1.0
    )
    
    print(f"Motion-based clips created: {len(motion_clips)}")
    
    print("Clip creator demo completed!")


def demo_summary_generator(video_path: str, output_dir: str):
    """Demonstrate summary generator functionality."""
    print("\n=== SUMMARY GENERATOR DEMO ===")
    
    generator = SummaryGenerator()
    
    try:
        # Generate comprehensive summary
        summary = generator.generate_comprehensive_summary(
            video_path,
            output_dir=output_dir
        )
        
        print("Comprehensive summary generated!")
        print(f"  Video: {summary.video_path}")
        print(f"  Duration: {summary.duration}s")
        print(f"  Activity Level: {summary.activity_level}")
        print(f"  Motion Percentage: {summary.motion_percentage:.2f}%")
        print(f"  Key Events: {len(summary.key_events)}")
        
        # Generate activity timeline
        timeline = generator.generate_activity_timeline(video_path, interval=2.0)
        print(f"Activity timeline generated with {len(timeline.get('segments', []))} segments")
        
        # Generate motion report
        motion_report = generator.generate_motion_report(video_path)
        if motion_report:
            stats = motion_report.get('motion_statistics', {})
            print(f"Motion report generated:")
            print(f"  Mean motion: {stats.get('mean_motion', 0):.2f}%")
            print(f"  Max motion: {stats.get('max_motion', 0):.2f}%")
        
        # Export reports
        json_report = generator.export_summary_report(summary, "json")
        html_report = generator.export_summary_report(summary, "html")
        txt_report = generator.export_summary_report(summary, "txt")
        
        print(f"Reports exported:")
        print(f"  JSON: {json_report}")
        print(f"  HTML: {html_report}")
        print(f"  Text: {txt_report}")
        
    except Exception as e:
        print(f"Summary generation failed: {e}")
        # Generate basic reports without visual summary
        timeline = generator.generate_activity_timeline(video_path, interval=2.0)
        motion_report = generator.generate_motion_report(video_path)
        
        print("Basic analysis completed:")
        print(f"  Timeline segments: {len(timeline.get('segments', []))}")
        print(f"  Motion report generated: {'Yes' if motion_report else 'No'}")
    
    print("Summary generator demo completed!")


def main():
    """Main demo function."""
    print("CCTV Video Analyzer - Video Module Demo")
    print("=" * 50)
    
    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create demo video
        video_path = os.path.join(temp_dir, "demo_video.mp4")
        create_demo_video(video_path, duration_seconds=10)
        
        # Create output directory for clips and summaries
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run all demos
        try:
            demo_video_player(video_path)
            demo_video_processor(video_path)
            demo_clip_creator(video_path, output_dir)
            demo_summary_generator(video_path, output_dir)
        except Exception as e:
            print(f"Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("Demo completed! All temporary files have been cleaned up.")
        
        # Show what files would have been created
        print("\nFiles that would be created in a real scenario:")
        print("- Video clips (MP4 format)")
        print("- Frame images (JPG format)")
        print("- Summary reports (JSON, HTML, TXT)")
        print("- Visual summaries (PNG charts)")


if __name__ == "__main__":
    main()