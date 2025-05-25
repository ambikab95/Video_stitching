import os
import cv2
import numpy as np
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import time

# Step 1: Extract video information using FFmpeg
def get_video_info(video_path):
    """Extract technical features of a video using FFmpeg"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name,duration,bits_per_raw_sample",
        "-show_entries", "format=size,format_name",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    info = json.loads(result.stdout)

    fps = info["streams"][0]["r_frame_rate"]
    if "/" in fps:
        num, den = map(int, fps.split("/"))
        fps = num / den
    else:
        fps = float(fps)

    size_bytes = float(info["format"]["size"])
    duration = float(info["streams"][0]["duration"])
    bitrate = (size_bytes * 8) / duration if duration > 0 else 0

    return {
        "width": int(info["streams"][0]["width"]),
        "height": int(info["streams"][0]["height"]),
        "fps": fps,
        "codec": info["streams"][0]["codec_name"],
        "duration": duration,
        "bitrate": bitrate,
        "file_size": size_bytes,
        "color_depth": int(info["streams"][0].get("bits_per_raw_sample", 8)),
        "file_format": info["format"]["format_name"]
    }

# Step 2: Normalize videos
def normalize_videos(video1_path, video2_path, output1_path, output2_path):
    """Normalize two videos to have the same technical specifications"""
    info1 = get_video_info(video1_path)
    info2 = get_video_info(video2_path)
    print("Original Video 1 Info:", info1)
    print("Original Video 2 Info:", info2)

    target_width = min(info1["width"], info2["width"])
    target_height = min(info1["height"], info2["height"])
    target_fps = min(info1["fps"], info2["fps"])
    target_codec = "h264"
    target_bitrate = min(info1["bitrate"], info2["bitrate"])

    for i, (input_path, output_path, info) in enumerate([
        (video1_path, output1_path, info1),
        (video2_path, output2_path, info2)
    ]):
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c:v", target_codec,
            "-b:v", f"{int(target_bitrate)}",
            "-r", f"{target_fps}",
            "-vf", f"scale={target_width}:{target_height}",
            "-pix_fmt", "yuv420p",
            "-y",
            output_path
        ]
        print(f"Normalizing video {i+1}...")
        subprocess.run(cmd, capture_output=True)

    norm_info1 = get_video_info(output1_path)
    norm_info2 = get_video_info(output2_path)
    print("Normalized Video 1 Info:", norm_info1)
    print("Normalized Video 2 Info:", norm_info2)

    return output1_path, output2_path

# Step 3: Zero-DCE Model
class ZeroDCE(nn.Module):
    def _init_(self):
        super(ZeroDCE, self)._init_()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(32, 24, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, enhancement_level=1.0):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        x_r = torch.tanh(self.conv7(x6))

        x_r = x_r * (enhancement_level * 0.5 + 0.5)
        r = x_r[:, :8, :, :]
        g = x_r[:, 8:16, :, :]
        b = x_r[:, 16:24, :, :]

        x_r = apply_curves(x[:, 0:1, :, :], r)
        x_g = apply_curves(x[:, 1:2, :, :], g)
        x_b = apply_curves(x[:, 2:3, :, :], b)

        return torch.cat([x_r, x_g, x_b], dim=1)

def apply_curves(x, curves):
    """Apply the 8 curves iteratively to the input"""
    enhanced = x.clone()
    for i in range(curves.shape[1]):
        alpha = curves[:, i:i+1, :, :] + 1
        enhanced = enhanced * alpha
    return enhanced

# Step 4: Load pretrained model
def load_pretrained_model(model_path=None):
    """Load pretrained Zero-DCE model"""
    model = ZeroDCE()
    if model_path is None:
        print("No model path provided. Using random weights.")
        return model
    try:
        print(f"Loading pretrained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Successfully loaded pretrained model!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using random weights as fallback.")
    return model

# Step 5: Enhance brightness
def enhance_brightness(input_video_path, output_video_path, model, enhancement_level=0.5):
    """Process a video using Zero-DCE model for brightness enhancement"""
    os.makedirs(os.path.dirname(output_video_path) or '.', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use temp file with mp4v codec for processing
    temp_output = output_video_path + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    print(f"Enhancing brightness for {input_video_path} with enhancement level {enhancement_level}...")
    with torch.no_grad():
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            input_tensor = torch.from_numpy(rgb_frame).float().permute(2, 0, 1).unsqueeze(0).to(device)
            enhanced_tensor = model(input_tensor, enhancement_level)
            enhanced_img = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_img = np.clip(enhanced_img * 255, 0, 255).astype(np.uint8)
            enhanced_frame = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
            out.write(enhanced_frame)

    cap.release()
    out.release()
    
    # Convert to web-compatible format
    cmd = [
        "ffmpeg",
        "-i", temp_output,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "faststart",
        "-y",
        output_video_path
    ]
    
    print(f"Converting to web-compatible format: {output_video_path}")
    subprocess.run(cmd, capture_output=True)
    
    # Remove the temporary file
    if os.path.exists(temp_output):
        os.remove(temp_output)
        
    return output_video_path

# Step 6: Direct brightness/contrast adjustment
def enhance_brightness_direct(input_video_path, output_video_path, brightness_factor=0, contrast_factor=1.0):
    """Enhance video brightness and contrast directly using OpenCV"""
    os.makedirs(os.path.dirname(output_video_path) or '.', exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use temp file with mp4v codec for processing
    temp_output = output_video_path + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    print(f"Enhancing brightness for {input_video_path} with brightness={brightness_factor}, contrast={contrast_factor}...")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=brightness_factor)
        out.write(adjusted)

    cap.release()
    out.release()
    
    # Convert to web-compatible format
    cmd = [
        "ffmpeg",
        "-i", temp_output,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "faststart",
        "-y",
        output_video_path
    ]
    
    print(f"Converting to web-compatible format: {output_video_path}")
    subprocess.run(cmd, capture_output=True)
    
    # Remove the temporary file
    if os.path.exists(temp_output):
        os.remove(temp_output)
    
    return output_video_path

# Step 7: Match color histogram
def match_color_histogram(source_frame, target_frame):
    """Match the color histogram of source_frame to target_frame"""
    source_hsv = cv2.cvtColor(source_frame, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)
    matched_source_hsv = source_hsv.copy()

    for i in range(3):
        src_hist = cv2.calcHist([source_hsv], [i], None, [256], [0, 256])
        tgt_hist = cv2.calcHist([target_hsv], [i], None, [256], [0, 256])
        src_cdf = src_hist.cumsum()
        src_cdf_normalized = src_cdf / src_cdf[-1]
        tgt_cdf = tgt_hist.cumsum()
        tgt_cdf_normalized = tgt_cdf / tgt_cdf[-1]
        lookup_table = np.zeros(256, dtype=np.uint8)
        for j in range(256):
            lookup_table[j] = np.argmin(np.abs(src_cdf_normalized[j] - tgt_cdf_normalized))
        matched_source_hsv[:, :, i] = lookup_table[source_hsv[:, :, i]]

    matched_source = cv2.cvtColor(matched_source_hsv, cv2.COLOR_HSV2BGR)
    return matched_source

# Step 8: Stitch videos seamlessly
def stitch_videos_seamlessly(video1_path, video2_path, output_path, blend_width=50):
    """Stitch two videos side by side with seamless blending"""
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Could not open one or both videos")

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    height = min(height1, height2)
    total_width = width1 + width2

    # Use a temp file with mp4v codec for processing
    temp_output = output_path + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps1, (total_width, height))

    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames1, total_frames2)

    blend_mask = np.zeros((height, total_width), dtype=np.float32)
    for i in range(blend_width):
        alpha = i / blend_width
        blend_mask[:, width1 - blend_width + i] = alpha
    inverse_blend_mask = 1.0 - blend_mask

    first_frame = True
    print("Stitching videos seamlessly...")
    for _ in tqdm(range(total_frames)):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        if frame1.shape[0] != height:
            frame1 = cv2.resize(frame1, (width1, height))
        if frame2.shape[0] != height:
            frame2 = cv2.resize(frame2, (width2, height))

        if first_frame:
            edge1 = frame1[:, -blend_width:]
            edge2 = frame2[:, :blend_width]
            frame2_matched = match_color_histogram(frame2, frame1)
            first_frame = False
        else:
            frame2_matched = match_color_histogram(frame2, frame1)

        stitched_frame = np.zeros((height, total_width, 3), dtype=np.float32)
        frame1_f32 = frame1.astype(np.float32)
        frame2_f32 = frame2_matched.astype(np.float32)

        for c in range(3):
            stitched_frame[:, :width1, c] = frame1_f32[:, :, c] * inverse_blend_mask[:, :width1]
            stitched_frame[:, width1:, c] = frame2_f32[:, :total_width-width1, c]
            overlap_width = min(blend_width, width1, width2)
            for i in range(overlap_width):
                pos1 = width1 - overlap_width + i
                pos2 = i
                alpha = i / overlap_width
                stitched_frame[:, pos1, c] = (1-alpha) * frame1_f32[:, pos1, c] + alpha * frame2_f32[:, pos2, c]

        stitched_frame = np.clip(stitched_frame, 0, 255).astype(np.uint8)
        out.write(stitched_frame)

    cap1.release()
    cap2.release()
    out.release()
    
    # Now convert the output to a web-compatible format using FFmpeg
    cmd = [
        "ffmpeg",
        "-i", temp_output,
        "-vcodec", "libx264",  # Use H.264 codec which is web compatible
        "-pix_fmt", "yuv420p", # Ensure pixel format is compatible
        "-movflags", "faststart", # Optimize for web streaming
        "-y",
        output_path
    ]
    
    print("Converting video to web-compatible format...")
    subprocess.run(cmd, capture_output=True)
    
    # Remove the temporary file
    if os.path.exists(temp_output):
        os.remove(temp_output)
        
    return output_path

# Step 9: List video files
def list_video_files(directory='.'):
    """List all video files in the specified directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(video_files)

# Step 10: Process and stitch videos
def process_and_stitch_videos_seamlessly(video1_path, video2_path, output_path, model_path=None,
                                         blend_width=50, enhancement_level=0.5,
                                         use_direct_enhancement=False, brightness=0, contrast=1.0):
    """Main function to process and seamlessly stitch two videos"""
    temp_dir = "temp_video_processing"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        print("Step 1: Normalizing video technical features...")
        normalized_video1 = os.path.join(temp_dir, "normalized_video1.mp4")
        normalized_video2 = os.path.join(temp_dir, "normalized_video2.mp4")
        normalize_videos(video1_path, video2_path, normalized_video1, normalized_video2)

        print("Step 2: Enhancing brightness...")
        enhanced_video1 = os.path.join(temp_dir, "enhanced_video1.mp4")
        enhanced_video2 = os.path.join(temp_dir, "enhanced_video2.mp4")

        if use_direct_enhancement:
            enhance_brightness_direct(normalized_video1, enhanced_video1, brightness, contrast)
            enhance_brightness_direct(normalized_video2, enhanced_video2, brightness, contrast)
        else:
            model = load_pretrained_model(model_path)
            enhance_brightness(normalized_video1, enhanced_video1, model, enhancement_level)
            enhance_brightness(normalized_video2, enhanced_video2, model, enhancement_level)

        print("Step 3: Stitching videos seamlessly...")
        stitch_videos_seamlessly(enhanced_video1, enhanced_video2, output_path, blend_width)

        # Verify the output file exists and is valid
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError(f"Failed to create output video: {output_path}")
            
        print(f"Video processing and seamless stitching complete! Output saved to: {output_path}")
        print(f"You can find the output at: {os.path.abspath(output_path)}")
        return output_path

    except Exception as e:
        print(f"Error during processing: {e}")
        raise e  # Re-raise to allow proper error handling in Streamlit

# Step 11: Main function
def main():
    video_files = list_video_files()
    if len(video_files) < 2:
        print(f"Not enough video files found in the current directory. Found: {len(video_files)}")
        print("Please ensure you have at least two video files in your project directory.")
        return

    print("Found the following video files in your directory:")
    for i, file in enumerate(video_files):
        print(f"{i+1}. {file}")

    try:
        print("\nSelect the first video (enter the number):")
        video1_idx = int(input()) - 1
        print("Select the second video (enter the number):")
        video2_idx = int(input()) - 1

        if video1_idx < 0 or video1_idx >= len(video_files) or video2_idx < 0 or video2_idx >= len(video_files):
            print("Invalid selection. Please run again and select valid video numbers.")
            return

        video1_path = video_files[video1_idx]
        video2_path = video_files[video2_idx]
        print(f"Selected videos:\n1. {video1_path}\n2. {video2_path}")
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    model_files = glob.glob('*.pth')
    model_path = None
    if model_files:
        print("\nFound the following model files:")
        for i, file in enumerate(model_files):
            print(f"{i+1}. {file}")
        print("\nDo you want to use one of these model files? (yes/no)")
        use_model = input().lower().strip()
        if use_model == 'yes':
            print("Select the model file (enter the number):")
            try:
                model_idx = int(input()) - 1
                if 0 <= model_idx < len(model_files):
                    model_path = model_files[model_idx]
                    print(f"Selected model: {model_path}")
                else:
                    print("Invalid selection. Proceeding without a model.")
            except ValueError:
                print("Invalid input. Proceeding without a model.")
    else:
        print("No .pth model files found in the current directory.")

    print("\nChoose enhancement method:")
    print("1. Zero-DCE model (AI-based enhancement)")
    print("2. Direct brightness/contrast adjustment")
    enhancement_choice = input("Enter choice (1/2): ").strip()
    use_direct_enhancement = enhancement_choice == "2"

    if use_direct_enhancement:
        print("\nEnter brightness adjustment (-50 to 50, 0 for no change):")
        brightness = 0
        try:
            brightness = float(input().strip())
        except:
            print("Using default brightness of 0")
        print("\nEnter contrast factor (0.5 to 1.5, 1.0 for no change):")
        contrast = 1.0
        try:
            contrast = float(input().strip())
        except:
            print("Using default contrast of 1.0")
    else:
        print("\nEnter enhancement level for Zero-DCE (0.1-1.0, lower = more subtle):")
        enhancement_level = 0.5
        try:
            enhancement_level = float(input().strip())
            if enhancement_level < 0.1:
                enhancement_level = 0.1
            elif enhancement_level > 1.0:
                enhancement_level = 1.0
        except:
            print(f"Using default enhancement level of {enhancement_level}")

    print("\nEnter blend width for seamless stitching (default 50 pixels):")
    blend_width = 50
    try:
        user_input = input().strip()
        if user_input:
            blend_width = int(user_input)
    except:
        print("Using default blend width of 50 pixels")

    timestamp = int(time.time())
    output_path = f"seamless_stitched_output_{timestamp}.mp4"

    if use_direct_enhancement:
        process_and_stitch_videos_seamlessly(
            video1_path, video2_path, output_path, model_path, blend_width,
            use_direct_enhancement=True, brightness=brightness, contrast=contrast
        )
    else:
        process_and_stitch_videos_seamlessly(
            video1_path, video2_path, output_path, model_path, blend_width,
            enhancement_level=enhancement_level
        )

    print(f"\nVideo processing complete!")
    print(f"Output video saved to: {os.path.abspath(output_path)}")

if __name__ == "_main_":
    main()