#!/usr/bin/env python3
"""
Universal AI Media Studio
A single clean interface for multiple AI image & video platforms.

Currently fully supported:
- xAI Grok Imagine (images + video)
- Kling AI (video focused)
- OpenAI DALL·E (images) + Sora placeholder

Easy to extend with: Runway ML, Luma Dream Machine, Pika Labs

Author: Grok
"""

import streamlit as st
import requests
import time
from datetime import datetime
from openai import OpenAI
from xai_sdk import Client as XAIClient
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Universal AI Media Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff9d, #00b8ff, #ff4b4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .platform-badge {
        font-size: 0.9rem;
        padding: 4px 12px;
        border-radius: 20px;
        background: #f0f2f6;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if "history" not in st.session_state:
    st.session_state.history = []
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("🎛️ Platform & Settings")

    platform = st.selectbox(
        "Select AI Platform",
        ["xAI Grok Imagine", "Kling AI", "OpenAI (DALL·E + Sora)"],
        index=0
    )

    st.divider()

    # API Key input (per platform)
    if platform == "xAI Grok Imagine":
        key_label = "xAI API Key"
        key_help = "Get from console.x.ai"
        env_var = "XAI_API_KEY"
    elif platform == "Kling AI":
        key_label = "Kling API Key (Bearer)"
        key_help = "Get from https://kling.ai/dev/api-key"
        env_var = "KLING_API_KEY"
    else:  # OpenAI
        key_label = "OpenAI API Key"
        key_help = "Get from platform.openai.com"
        env_var = "OPENAI_API_KEY"

    api_key = st.text_input(
        key_label,
        type="password",
        value=st.session_state.api_keys.get(platform, os.getenv(env_var, "")),
        help=key_help
    )

    if api_key:
        st.session_state.api_keys[platform] = api_key
        st.success("API Key saved for this session")
    else:
        st.warning("Please enter your API key")

    st.divider()
    st.caption("💡 Tip: Keys are only stored in your browser session. They are never saved on any server.")

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🎨 Universal AI Media Studio</h1>', unsafe_allow_html=True)
st.markdown(f'<div class="platform-badge">Currently using: <strong>{platform}</strong></div>', unsafe_allow_html=True)

# ==================== PLATFORM CONFIG ====================
PLATFORM_CONFIG = {
    "xAI Grok Imagine": {
        "supports_image": True,
        "supports_video": True,
        "image_models": ["grok-imagine-image-quality", "grok-imagine-image"],
        "video_models": ["grok-imagine-video", "grok-imagine-video-1.5"],
    },
    "Kling AI": {
        "supports_image": False,
        "supports_video": True,
        "video_models": ["kling-v3", "kling-v2-6", "kling-v2.6-pro"],
    },
    "OpenAI (DALL·E + Sora)": {
        "supports_image": True,
        "supports_video": True,  # Sora placeholder
        "image_models": ["dall-e-3", "dall-e-2"],
        "video_models": ["sora"],  # Placeholder
    }
}

config = PLATFORM_CONFIG[platform]

# ==================== GENERATION FUNCTIONS ====================

def generate_grok_image(prompt, model, aspect_ratio, resolution, num_images, api_key):
    client = XAIClient(api_key=api_key)
    if num_images > 1:
        responses = client.image.sample_batch(
            prompt=prompt, model=model, n=num_images,
            aspect_ratio=aspect_ratio if aspect_ratio != "auto" else None,
            resolution=resolution
        )
        return [r.url for r in responses]
    else:
        response = client.image.sample(
            prompt=prompt, model=model,
            aspect_ratio=aspect_ratio if aspect_ratio != "auto" else None,
            resolution=resolution
        )
        return [response.url]


def generate_grok_video(prompt, model, duration, aspect_ratio, resolution, api_key):
    client = XAIClient(api_key=api_key)
    response = client.video.generate(
        prompt=prompt, model=model, duration=duration,
        aspect_ratio=aspect_ratio, resolution=resolution
    )
    return response.url


def generate_kling_video(prompt, model, duration, aspect_ratio, mode, sound, api_key, region="Beijing"):
    base_url = "https://api-beijing.klingai.com" if region == "Beijing" else "https://api-singapore.klingai.com"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model_name": model,
        "prompt": prompt,
        "duration": str(duration),
        "aspect_ratio": aspect_ratio,
        "mode": mode,
        "sound": "on" if sound else "off"
    }

    # Create task
    resp = requests.post(f"{base_url}/v1/videos/text2video", headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Kling API error: {resp.text}")
    task_id = resp.json().get("task_id") or resp.json().get("id")

    # Poll
    poll_url = f"{base_url}/v1/videos/text2video/{task_id}"
    for _ in range(120):  # ~8 minutes max
        time.sleep(4)
        poll_resp = requests.get(poll_url, headers=headers, timeout=15)
        if poll_resp.status_code == 200:
            data = poll_resp.json()
            status = data.get("status") or data.get("task_status")
            if status in ["succeed", "success", "completed"]:
                # Extract video URL (try common paths)
                video_url = None
                if "videos" in data and data["videos"]:
                    video_url = data["videos"][0].get("url")
                elif "data" in data and "videos" in data["data"]:
                    video_url = data["data"]["videos"][0].get("url")
                if video_url:
                    return video_url
            elif status in ["failed", "error"]:
                raise Exception(f"Kling generation failed: {data}")
    raise Exception("Kling video generation timed out")


def generate_openai_image(prompt, model, size, quality, n, api_key):
    client = OpenAI(api_key=api_key)
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n
    )
    return [img.url for img in response.data]


# ==================== MAIN UI ====================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✍️ Prompt")
    prompt = st.text_area(
        "Describe what you want to generate",
        height=120,
        placeholder="A cyberpunk cat riding a flying skateboard through neon Tokyo at night..."
    )

with col2:
    st.subheader("⚙️ Generation Type")
    gen_type = st.radio(
        "What do you want to generate?",
        options=["Image", "Video"] if config["supports_video"] else ["Image"],
        horizontal=True
    )

st.divider()

# ==================== DYNAMIC PARAMETERS ====================
st.subheader("Advanced Options")

if platform == "xAI Grok Imagine":
    if gen_type == "Image":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            model = st.selectbox("Model", config["image_models"], index=0)
        with c2:
            aspect = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "4:3", "auto"], index=0)
        with c3:
            resolution = st.selectbox("Resolution", ["1k", "2k"], index=1)
        with c4:
            num_images = st.slider("Number of images", 1, 4, 1)

    else:  # Video
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            model = st.selectbox("Video Model", config["video_models"], index=0)
        with c2:
            duration = st.select_slider("Duration (sec)", [5, 8, 10, 12], value=8)
        with c3:
            aspect = st.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1"], index=0)
        with c4:
            resolution = st.selectbox("Resolution", ["480p", "720p"], index=1)

elif platform == "Kling AI":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        model = st.selectbox("Model", config["video_models"], index=0)
    with c2:
        duration = st.select_slider("Duration (sec)", [5, 10], value=5)
    with c3:
        aspect = st.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1"], index=0)
    with c4:
        mode = st.radio("Quality Mode", ["std", "pro"], horizontal=True, index=1)
    sound = st.checkbox("Generate audio/sound", value=True)

else:  # OpenAI
    if gen_type == "Image":
        c1, c2, c3 = st.columns(3)
        with c1:
            model = st.selectbox("Model", config["image_models"], index=0)
        with c2:
            size = st.selectbox("Size", ["1024x1024", "1792x1024", "1024x1792"], index=0)
        with c3:
            quality = st.selectbox("Quality", ["standard", "hd"], index=1)
        num_images = st.slider("Number of images", 1, 4, 1)
    else:
        st.info("Sora video generation is available via OpenAI API (enterprise access in some regions). Contact OpenAI for access.")
        st.stop()

# ==================== GENERATE BUTTON ====================
generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True, disabled=not api_key)

if generate_btn and prompt.strip():
    api_key = st.session_state.api_keys.get(platform)

    with st.spinner(f"Generating with {platform}..."):
        try:
            if platform == "xAI Grok Imagine":
                if gen_type == "Image":
                    urls = generate_grok_image(prompt, model, aspect, resolution, num_images, api_key)
                    result_type = "image"
                else:
                    url = generate_grok_video(prompt, model, duration, aspect, resolution, api_key)
                    urls = [url]
                    result_type = "video"

            elif platform == "Kling AI":
                url = generate_kling_video(prompt, model, duration, aspect, mode, sound, api_key)
                urls = [url]
                result_type = "video"

            else:  # OpenAI
                if gen_type == "Image":
                    urls = generate_openai_image(prompt, model, size, quality, num_images, api_key)
                    result_type = "image"
                else:
                    st.error("Sora video not yet implemented in this demo. Use the official OpenAI playground or API directly.")
                    st.stop()

            # Save to history
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "platform": platform,
                "type": result_type,
                "prompt": prompt,
                "urls": urls,
                "model": model if 'model' in locals() else "default"
            })

            # Display results
            st.success("✅ Generation complete!")

            if result_type == "image":
                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    with cols[i % 2]:
                        st.image(url, caption=f"Result {i+1}", use_container_width=True)
                        st.markdown(f"[⬇️ Download]({url})")
            else:
                st.video(urls[0])
                st.markdown(f"[⬇️ Download Video]({urls[0]})")

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

# ==================== HISTORY ====================
st.divider()
st.subheader("📜 Generation History")

if not st.session_state.history:
    st.info("No generations yet. Create something above!")
else:
    for item in reversed(st.session_state.history):
        with st.expander(f"{item['platform']} • {item['type'].upper()} • {item['timestamp']}", expanded=False):
            st.write(f"**Prompt:** {item['prompt']}")
            if item['type'] == "image":
                for url in item['urls']:
                    st.image(url, width=300)
            else:
                st.video(item['urls'][0])

# ==================== EXTENSIBILITY GUIDE ====================
with st.expander("🛠️ How to Add More Platforms (Runway, Luma, Pika)"):
    st.markdown("""
    **Adding a new platform is straightforward.** Here's the pattern:

    1. Add the platform name to the `selectbox`
    2. Add its config in `PLATFORM_CONFIG`
    3. Create a `generate_xxx()` function (use `requests` or official SDK)
    4. Add the logic inside the `if generate_btn` block

    **Example for Runway ML (very similar to Kling):**
    ```python
    from runwayml import RunwayML
    client = RunwayML(api_key=api_key)
    task = client.text_to_video.create(
        model="gen4.5",
        promptText=prompt,
        ratio="1280:720",
        duration=5
    ).waitForTaskOutput()
    video_url = task.output[0]
    ```

    **Luma Dream Machine** uses their official `lumaai` Python SDK (very clean).

    **Pika Labs** has good API support via their official endpoint or through fal.ai.

    Want me to add any of these three right now? Just say the word.
    """)

st.caption("This is a personal tool. API keys are handled securely in your session. Not affiliated with any of the platforms.")