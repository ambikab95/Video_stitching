import os
import sys
import time
import glob
import subprocess
import tempfile
import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import base64

# Import the backend functions from your existing code
# This assumes your backend code file is in the same directory
from paste import (
    ZeroDCE, load_pretrained_model, get_video_info, normalize_videos, 
    enhance_brightness, enhance_brightness_direct, stitch_videos_seamlessly,
    process_and_stitch_videos_seamlessly
)

st.set_page_config(
    page_title="Video Stitching Tool",
    page_icon="🎬",
    layout="wide",
)

st.title("🎞 Video Stitching & Enhancement Tool")
st.write("Upload two videos to stitch them side by side with seamless blending")

# Create temp folder for processing
TEMP_DIR = "temp_streamlit_videos"
os.makedirs(TEMP_DIR, exist_ok=True)

# Function to save uploaded file
def save_uploaded_file(uploaded_file, folder="temp_streamlit_videos"):
    """Save an uploaded file to the specified folder"""
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to display video
# In your app.py file, update the display_video function:

def display_video(video_path, caption=""):
    """Display video in Streamlit with proper encoding"""
    if os.path.exists(video_path):
        # Ensure the video is correctly encoded for web display
        web_compatible_path = video_path
        if not video_path.endswith("_web.mp4"):
            web_compatible_path = f"{os.path.splitext(video_path)[0]}_web.mp4"
            # Convert to web-compatible format if not already done
            if not os.path.exists(web_compatible_path):
                cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "faststart",
                    "-y",
                    web_compatible_path
                ]
                try:
                    subprocess.run(cmd, capture_output=True)
                except Exception as e:
                    st.warning(f"Error preparing video for display: {e}")
                    web_compatible_path = video_path
        
        # Display the video
        video_file = open(web_compatible_path, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
        st.caption(caption)
    else:
        st.warning(f"Video file not found: {video_path}") 

# Function to get video thumbnail
def get_video_thumbnail(video_path):
    """Extract a thumbnail from the video"""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# Create tabs
tab1, tab2, tab3 = st.tabs(["Upload & Preview", "Process Settings", "Output"])

# Tab 1: Upload & Preview
with tab1:
    st.header("Upload Videos")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Video")
        video1_file = st.file_uploader("Upload first video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if video1_file:
            video1_path = save_uploaded_file(video1_file)
            st.session_state.video1_path = video1_path
            st.success(f"✅ {video1_file.name} uploaded successfully")
            
            # Show video info
            with st.expander("Video 1 Technical Info"):
                try:
                    info = get_video_info(video1_path)
                    st.json(info)
                except Exception as e:
                    st.error(f"Error getting video info: {e}")
            
            # Show video preview
            display_video(video1_path, "Video 1 Preview")
    
    with col2:
        st.subheader("Second Video")
        video2_file = st.file_uploader("Upload second video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if video2_file:
            video2_path = save_uploaded_file(video2_file)
            st.session_state.video2_path = video2_path
            st.success(f"✅ {video2_file.name} uploaded successfully")
            
            # Show video info
            with st.expander("Video 2 Technical Info"):
                try:
                    info = get_video_info(video2_path)
                    st.json(info)
                except Exception as e:
                    st.error(f"Error getting video info: {e}")
            
            # Show video preview
            display_video(video2_path, "Video 2 Preview")

# Tab 2: Process Settings
with tab2:
    st.header("Processing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enhancement Settings")
        enhancement_method = st.radio(
            "Choose enhancement method:",
            ["Zero-DCE model (AI-based enhancement)", "Direct brightness/contrast adjustment"],
            index=0
        )
        
        use_direct_enhancement = enhancement_method == "Direct brightness/contrast adjustment"
        
        if use_direct_enhancement:
            brightness = st.slider("Brightness adjustment", -50, 50, 0, help="Adjust the brightness level")
            contrast = st.slider("Contrast factor", 0.5, 1.5, 1.0, 0.1, help="Adjust the contrast level")
        else:
            enhancement_level = st.slider("Zero-DCE Enhancement Level", 0.1, 1.0, 0.5, 0.1, 
                                          help="Lower values are more subtle, higher values give stronger enhancement")

    with col2:
        st.subheader("Stitching Settings")
        blend_width = st.slider("Blend Width (pixels)", 10, 200, 50, 
                                help="Width of the blending region between the two videos")
        
        # Model upload for Zero-DCE
        if not use_direct_enhancement:
            st.subheader("Zero-DCE Model (Optional)")
            model_file = st.file_uploader("Upload pretrained model (.pth file)", type=["pth"])
            if model_file:
                model_path = save_uploaded_file(model_file)
                st.session_state.model_path = model_path
                st.success(f"✅ Model {model_file.name} uploaded successfully")
            else:
                st.session_state.model_path = None
                st.info("No model uploaded, will use random weights")
    
    # Process button
    st.subheader("Start Processing")
    process_button = st.button("🚀 Process and Stitch Videos", type="primary", use_container_width=True)
    
    if process_button:
        if not hasattr(st.session_state, 'video1_path') or not hasattr(st.session_state, 'video2_path'):
            st.error("Please upload both videos first")
        else:
            timestamp = int(time.time())
            output_path = os.path.join(TEMP_DIR, f"seamless_stitched_output_{timestamp}.mp4")
            st.session_state.output_path = output_path
            
            with st.spinner("Processing videos... This may take several minutes."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Update progress
                    status_text.text("Step 1/3: Normalizing videos...")
                    progress_bar.progress(10)
                    
                    # Step 2: Update progress
                    status_text.text("Step 2/3: Enhancing brightness...")
                    progress_bar.progress(40)
                    
                    # Process videos based on enhancement method
                    if use_direct_enhancement:
                        process_and_stitch_videos_seamlessly(
                            st.session_state.video1_path, 
                            st.session_state.video2_path, 
                            output_path,
                            None,  # No model path needed
                            blend_width,
                            use_direct_enhancement=True,
                            brightness=brightness,
                            contrast=contrast
                        )
                    else:
                        model_path = getattr(st.session_state, 'model_path', None)
                        process_and_stitch_videos_seamlessly(
                            st.session_state.video1_path, 
                            st.session_state.video2_path, 
                            output_path,
                            model_path,
                            blend_width,
                            enhancement_level=enhancement_level
                        )
                    
                    # Final step
                    status_text.text("Step 3/3: Stitching videos...")
                    progress_bar.progress(80)
                    
                    # Complete
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    st.success("✅ Videos processed and stitched successfully!")
                    st.session_state.processing_complete = True
                    
                except Exception as e:
                    st.error(f"Error during processing: {e}")
                    st.session_state.processing_complete = False

# Tab 3: Output
with tab3:
    st.header("Output Video")
    
    if hasattr(st.session_state, 'processing_complete') and st.session_state.processing_complete:
        if hasattr(st.session_state, 'output_path') and os.path.exists(st.session_state.output_path):
            st.subheader("Processed and Stitched Video")
            display_video(st.session_state.output_path, "Final Stitched Video")
            
            # Download button
            with open(st.session_state.output_path, "rb") as file:
                btn = st.download_button(
                    label="📥 Download Stitched Video",
                    data=file,
                    file_name=os.path.basename(st.session_state.output_path),
                    mime="video/mp4",
                    use_container_width=True
                )
                
            # Video info
            with st.expander("Output Video Technical Info"):
                try:
                    info = get_video_info(st.session_state.output_path)
                    st.json(info)
                except Exception as e:
                    st.error(f"Error getting video info: {e}")
        else:
            st.warning("Output video file not found. Please check the processing logs.")
    else:
        st.info("No processed video yet. Go to the 'Process Settings' tab and click 'Process and Stitch Videos'.")

st.sidebar.title("Help & Info")

with st.sidebar.expander("📋 Quick Guide", expanded=True):
    st.markdown("""
    1. *Upload Videos*: Upload two videos on the first tab
    2. *Configure Settings*: Choose enhancement method and stitching parameters
    3. *Process*: Click the process button to start stitching
    4. *Download*: View and download the final stitched video
    """)

with st.sidebar.expander("⚙ About Enhancement Methods"):
    st.markdown("""
    - *Zero-DCE*: AI-based enhancement for low-light videos
    - *Direct Adjustment*: Manual brightness/contrast control
    """)

with st.sidebar.expander("🔧 Tips"):
    st.markdown("""
    - Similar resolution videos work best
    - For optimal stitching, use videos with similar lighting conditions
    - Larger blend width creates smoother transitions but may cause more blurring
    - Processing time depends on video length and resolution
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Video Processing & Stitching Tool")