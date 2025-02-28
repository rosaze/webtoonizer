import streamlit as st
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI
import logging
from clip_analyzer import CLIPAnalyzer
from datetime import datetime
from PIL import Image
from general_text_input import TextToWebtoonConverter
from io import BytesIO
from image_gen import generate_image_from_text
from save_utils import save_session

@dataclass
class NonFictionConfig:
    style: str
    visualization_type: str
    aspect_ratio: str
    num_images: int
    emphasis: str = "clarity"

class NonFictionConverter:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.clip_analyzer = CLIPAnalyzer()
        self.setup_logging()

        self.visualization_types = {
           "Explain": {
                "prompt": "simple minimalistic shapes, thin and sharp lines, clean composition, no text",
                "layout": "minimalistic single-concept layout",
                "elements": "sole object, no unnecessary shading or details",
                "style": "educational minimalistic style with thin lines"
           },
           "Compare": {
                "prompt": "two-column comparison, thin outlines, minimalistic shapes, clean layout, no unnecessary details",
                "layout": "side-by-side layout, focus on clear differences",
                "elements": "precise shapes, no shading, no text",
                "style": "minimalistic cartoon style with fine lines"
           },
           "Show Process": {
                "prompt": "step-by-step flow, clean lines, thin minimalistic shapes, cartoon-like simplicity without exaggeration",
                "layout": "horizontal or vertical progression with arrows",
                "elements": "single-colored shapes, no gradients, no text",
                "style": "thin line cartoon minimalistic style"
           },
           "Explain Principle": {
                "prompt": "cause-and-effect diagram with minimalistic shapes, thin lines, plain white background, no text",
                "layout": "input-output or cause-effect structure",
                "elements": "clear, distinct shapes, no complex details",
                "style": "scientific minimalistic style with cartoon simplicity"
           }
        }

    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def split_content_into_scenes(self, text: str, num_scenes: int) -> List[str]:
        """Split text into explainable scenes"""
        try:
            prompt = f"""Split the following content into {num_scenes} key scenes.
            Each scene should be visually representable.

            Analysis criteria:
            1. Focus on key concepts or main ideas
            2. Prioritize content that is easy to visualize
            3. Maintain logical flow of the content
            4. Reconstruct complex content into simple relationships
            5. Convert abstract ideas into concrete metaphors

            Current text:
            {text}

            Each scene should be formatted as follows:
            - Focus on visual elements
            - Clearly explain relationships and structure
            - Describe in 1-2 sentences per scene"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )

            scenes = response.choices[0].message.content.strip().split("\n\n")
            return [scene.strip() for scene in scenes if scene.strip()][:num_scenes]
        
        except Exception as e:
            logging.error(f"Scene splitting failed: {str(e)}")
            return [text]

    def create_scene_description(self, scene: str, config: NonFictionConfig) -> str:
        """Convert scene to a webtoon-style prompt"""
        try:
            if config.visualization_type not in self.visualization_types:
                logging.error(f"Invalid visualization type: {config.visualization_type}")
                vis_type = self.visualization_types["Explain"]
            else:
                vis_type = self.visualization_types[config.visualization_type]

            max_length = 200
            content = scene[:max_length] if len(scene) > max_length else scene

            prompt = f"""Create a clear, simple educational illustration:

Main concept: {content}

Style requirements:
- {vis_type['style']}
- Layout: {vis_type['layout']}
- Elements: {vis_type['elements']}
- Visual style: {vis_type['prompt']}
- Single focused concept per image
- Bold, clean lines like manhwa/manga style
- Soft, pleasant color palette (2-3 colors maximum)
- White or very light background

Must include:
- One clear focal point
- Simple visual metaphor
- Easy-to-understand layout
- Gentle, rounded edges
- Ample white space around main element

Must avoid:
- Multiple competing concepts
- Complex diagrams or flowcharts
- Technical symbols or formulas
- Connecting lines or arrows
- Text labels or numbers
- Cluttered compositions
- Multiple scenes in one image
"""

            return prompt

        except Exception as e:
            logging.error(f"Scene description creation failed: {str(e)}")
            raise

    def process_submission(self, text: str, config: NonFictionConfig):
        """Generate educational content in webtoon style"""
        try:
            progress_bar = st.progress(0)
            status = st.empty()

            if 'generation_logs' not in st.session_state:
                st.session_state.generation_logs = []

            start_time = datetime.now()

            st.sidebar.markdown("### 🔍 CLIP Analyzer Info")
            st.sidebar.info(f"Device: {self.clip_analyzer.device}")
            st.sidebar.info(f"Model: openai/clip-vit-base-patch32")

            status.info("📝 Analyzing content...")
            scenes = self.split_content_into_scenes(text, config.num_images)
            progress_bar.progress(0.2)

            generated_images = {}
            scene_descriptions = []

            generation_metrics = {
                'total_time': 0,
                'avg_clip_score': 0,
                'scores': [],
                'generation_attempts': []
            }

            for i, scene in enumerate(scenes):
                scene_start_time = datetime.now()
                status.info(f"🎨 Generating scene {i+1}/{len(scenes)}...")

                prompt = self.create_scene_description(scene, config)
                scene_descriptions.append(prompt)

                image_url, _, _ = generate_image_from_text(
                    prompt=prompt,
                    style="minimalistic",
                    aspect_ratio=config.aspect_ratio,
                    negative_prompt=(
                        "abstract art, messy layout, unclear connections, "
                        "photorealistic style, 3d rendering, "
                        "complex textures, dark colors, "
                        "artistic interpretation, painterly style"
                    )
                )

                if image_url:
                    generated_images[i] = image_url

                    quality_check = self.clip_analyzer.validate_image(
                        image_url, 
                        prompt,
                        return_score=True
                    )

                    if i % 2 == 0:
                        cols = st.columns(min(2, config.num_images - i))

                    with cols[i % 2]:
                        st.image(image_url, use_column_width=True)

                        with st.expander("🔍 CLIP Analysis Results", expanded=False):
                            col1, col2 = st.columns(2)
                            score = quality_check.get("similarity_score", 0.0)

                            with col1:
                                st.metric("Quality Score", f"{score:.2f}")
                            with col2:
                                if score >= 0.7:
                                    st.success("✓ High Quality")
                                elif score >= 0.5:
                                    st.warning("△ Medium Quality")
                                else:
                                    st.error("⚠ Low Quality")

                            st.write("Prompt Matching:")
                            st.progress(score)

                            scene_time = (datetime.now() - scene_start_time).total_seconds()
                            st.info(f"⏱ Generation Time: {scene_time:.1f}s")

                        summary = self.summarize_scene(scene)
                        st.markdown(
                            f"<p style='text-align: center; font-size: 14px;'>{summary}</p>", 
                            unsafe_allow_html=True
                        )

                    generation_metrics['scores'].append(score)
                    generation_metrics['generation_attempts'].append({
                        'scene_number': i + 1,
                        'clip_score': score,
                        'generation_time': scene_time
                    })

                progress_bar.progress((i + 1) / config.num_images)

            generation_metrics['total_time'] = (datetime.now() - start_time).total_seconds()
            generation_metrics['avg_clip_score'] = sum(generation_metrics['scores']) / len(generation_metrics['scores'])

            st.session_state.generation_logs.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'config': config.__dict__,
                'metrics': generation_metrics
            })

            st.sidebar.markdown("### 📊 Summary")
            st.sidebar.metric("Average CLIP Score", f"{generation_metrics['avg_clip_score']:.2f}")
            st.sidebar.metric("Total Generation Time", f"{generation_metrics['total_time']:.1f}s")

            st.session_state.generated_images = generated_images
            st.session_state.scene_descriptions = scene_descriptions

            status.success("✨ Webtoon generation complete!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error in process_submission: {str(e)}")

    def summarize_scene(self, description: str) -> str:
        """Generate a description summarizing the scene context and meaning"""
        try:
            prompt = f"""Describe the context and main message of the following visual explanation:

Requirements:
1. Avoid simple descriptions like "scene of~"; include context and meaning.
2. Use present tense if possible.
3. Include causality or changes if applicable.
4. Keep it within 70 characters.

Example:
❌ "A scene with a circle and arrows."
✔️ "Shows water turning into vapor in a cycle."

Description to summarize:
{description}"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": description}
                ],
                temperature=0.7,
                max_tokens=200
            )

            summary = response.choices[0].message.content.strip()
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
            return summary[:150]

        except Exception as e:
            logging.error(f"Scene summarization failed: {str(e)}")
            return description[:70]

    def render_ui(self):
        st.title("Educational/Scientific Text Visualization")
        if 'generated_images' not in st.session_state:
            st.session_state.generated_images = {}
            st.session_state.current_config = None
            st.session_state.current_text = None
            st.session_state.scene_descriptions = []

        input_method = st.radio(
            "Select Input Method",
            ["Direct Input", "File Upload"],
            horizontal=True
        )

        with st.form("nonfiction_input_form"):
            text_content = None
            if input_method == "Direct Input":
                text_content = st.text_area(
                    "Enter the content you want to explain",
                    placeholder="We'll help you explain difficult concepts easily.",
                    height=200
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload a file",
                    type=['txt', 'pdf', 'docx', 'doc'],
                    help="Supported formats: TXT, PDF, DOCX"
                )
                if uploaded_file:
                    text_content = TextToWebtoonConverter.read_file_content(uploaded_file)
                    if text_content:
                        st.success("File uploaded successfully!")
                        with st.expander("View file content"):
                            st.text(text_content[:500] + "..." if len(text_content) > 500 else text_content)

            col1, col2 = st.columns(2)

            with col1:
                visualization_type = st.selectbox(
                    "How would you like to explain?",
                    list(self.visualization_types.keys()),
                    help="Choose the most suitable explanation style for the content"
                )

                num_images = st.radio(
                    "How many illustrations do you need?",
                    options=[1, 2, 3, 4],
                    horizontal=True
                )

            with col2:
                aspect_ratio = st.selectbox(
                    "Image Aspect Ratio",
                    ["Square (1:1)", "Wide (16:9)", "Portrait (9:16)"]
                )

            submit = st.form_submit_button("✨ Start Generating Webtoon")

            if submit and text_content:
                ratio_map = {
                    "Square (1:1)": "1:1",
                    "Wide (16:9)": "16:9",
                    "Portrait (9:16)": "9:16"
                }

                config = NonFictionConfig(
                    style="webtoon",
                    visualization_type=visualization_type,
                    aspect_ratio=ratio_map.get(aspect_ratio, "1:1"),
                    num_images=num_images,
                    emphasis="clarity"
                )

                st.session_state.current_config = config
                st.session_state.current_text = text_content
                self.process_submission(text_content, config)

            elif submit:
                st.warning("Please enter text or upload a file!")

        if st.session_state.generated_images:
            if st.button("💾 Save Current Session"):
                save_config = {
                    'type': 'education',
                    'title': st.session_state.current_text[:100],
                    'text': st.session_state.current_text,
                    'visualization_type': st.session_state.current_config.visualization_type,
                    'aspect_ratio': st.session_state.current_config.aspect_ratio,
                    'num_images': st.session_state.current_config.num_images,
                    'scene_descriptions': st.session_state.scene_descriptions
                }
                session_dir = save_session(save_config, st.session_state.generated_images)

                st.success(f"✅ Successfully saved! Location: {session_dir}")

def main():
    st.set_page_config(
        page_title="Educational Webtoon Creator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        client = OpenAI()
        converter = NonFictionConverter(client)
        converter.render_ui()
    except Exception as e:
        st.error(f"Error occurred while running the application: {e}")

if __name__ == "__main__":
    main()
