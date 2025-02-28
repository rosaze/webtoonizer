import streamlit as st
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import logging
from openai import OpenAI
import torch
from io import BytesIO
import PyPDF2
from clip_analyzer import CLIPAnalyzer
from docx import Document
from image_gen import generate_image_from_text
from save_utils import save_session
@dataclass
class SceneConfig:
    style: str
    composition: str
    mood: str
    character_desc: str
    aspect_ratio: str

class TextToWebtoonConverter:
    def __init__(self, openai_client: OpenAI, clip_analyzer):
        self.client = openai_client
        self.clip_analyzer = clip_analyzer
        self.setup_logging()
        self.style_guides = {
            "Minimalist": {
                "prompt": "minimal details, simple lines, clean composition, essential elements only",
                "emphasis": "Focus on simplicity and negative space"
            },
            "Pictogram": {
                "prompt": "pictogram style, symbolic representation, simplified shapes, icon-like style",
                "emphasis": "Clear silhouettes and symbolic elements"
            },
            "Cartoon": {
                "prompt": "animated style, exaggerated features, bold colors",
                "emphasis": "Expressive and dynamic elements"
            },
            "Webtoon": {
                "prompt": "webtoon style, manhwa art style, clean lines, vibrant colors",
                "emphasis": "Dramatic angles and clear storytelling"
            },
            "Artistic": {
                "prompt": "painterly style, artistic interpretation, creative composition",
                "emphasis": "Atmospheric and textural details"
            }
        }
        
        self.mood_guides = {
            "Casual": {
                "prompt": "natural lighting, soft colors, everyday atmosphere",
                "lighting": "warm, natural daylight",
                "color": "neutral, balanced palette"
            },
            "Tense": {
                "prompt": "dramatic lighting, high contrast, intense atmosphere",
                "lighting": "harsh shadows, dramatic highlights",
                "color": "high contrast, intense tones"
            },
            "Serious": {
                "prompt": "subdued lighting, serious atmosphere, formal composition",
                "lighting": "soft, directional light",
                "color": "muted, serious tones"
            },
            "Warm": {
                "prompt": "warm colors, soft lighting, comfortable atmosphere",
                "lighting": "golden hour, soft glow",
                "color": "warm, inviting palette"
            },
            "Joyful": {
                "prompt": "bright lighting, warm colors, dynamic composition",
                "lighting": "bright, cheerful",
                "color": "vibrant, playful colors"
            }
        }
        
        self.composition_guides = {
            "Background and Character": "balanced composition of character and background, eye-level shot",
            "Close-up Shot": "close-up shot, focused on character's expression",
            "Interactive": "two-shot composition, characters facing each other",
            "Landscape-focused": "wide shot, emphasis on background scenery",
            "Default": "standard view, balanced composition"
        }
         # 부정적 조건을 클래스 속성으로 정의
        self.negative_elements = (
            "blurry images, distorted faces, text in image, unrealistic proportions, "
            "extra limbs, overly complicated backgrounds, too much characters,excessive details,poor lighting, bad anatomy, "
            "abstract images, cut-off elements"
        )

    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    @staticmethod
    def read_file_content(uploaded_file):
        """다양한 형식의 파일 읽기"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
        
            if file_extension == 'txt':
                bytes_data = uploaded_file.getvalue()
                encoding = 'utf-8'
                try:
                    return bytes_data.decode(encoding)
                except UnicodeDecodeError:
                    return bytes_data.decode('cp949')
            
            elif file_extension == 'pdf':
                pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            
            elif file_extension in ['docx', 'doc']:
                doc = Document(BytesIO(uploaded_file.getvalue()))
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            else:
                return None
            
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")
            return None

    def analyze_text(self, text: str, cut_count: int) -> List[str]:
        """텍스트를 분석하여 주요 장면들을 추출"""
        try:
            system_prompt = """From the perspective of a webtoon artist, select scenes based on the following criteria:
             1. Moments with strong visual impact
            2. Scenes where the character's emotions are heightened
            3. Turning points in the story
            4. Compositions that can enhance reader immersion
            5. Scenes with a natural flow of sequential cuts"""    
            
            user_prompt = f"""Select the {cut_count} most suitable scenes for webtoon adaptation from the following text.
            Each scene must include the following elements:
               - Detailed spatial sense and background description
                - Character actions and expressions
                - Lighting and atmosphere
                - Visual focal points
                - Connection with preceding and following scenes
                
            text
            {text}"""
             # 메시지 데이터
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
            st.subheader("🔍 GPT Request Message")
            st.text_area("Request Messages", value=f"{messages}", height=200)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            scenes = response.choices[0].message.content.strip().split("\n\n")
            return scenes[:cut_count]
            
        except Exception as e:
            logging.error(f"Scene analysis failed: {str(e)}")
            raise

    def analyze_story_by_cuts(self, text: str, cut_count: int) -> Dict[str, str]:
        """컷 수에 따른 스토리 분석"""
        try:
            scene_types = {
                1: ["Key Scene"],
                2: ["Introduction", "Climax"],
                3: ["Beginning", "Development", "Conclusion"],
                4: ["기(起)", "승(承)", "전(轉)", "결(結)"]
            }
            
            prompt = f"""Please divide the following story into {cut_count} key scenes and analyze them.
            Each scene should follow the following structure:
            {scene_types[cut_count]}
            
            Each scene must include the following elements:
            - Detailed spatial sense and background description
            - Character actions and expressions
            - Lighting and atmosphere
            - Visual focal points
            - Connection with preceding and following scenes

            Text:
            {text}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            scenes = {}
            raw_scenes = response.choices[0].message.content.strip().split("\n\n")
            
            for scene_type, scene in zip(scene_types[cut_count], raw_scenes):
                scenes[scene_type] = scene
            
            return scenes
            
        except Exception as e:
            logging.error(f"Scene analysis failed: {str(e)}")
            raise

    @staticmethod
    def get_image_size(aspect_ratio: str) -> str:
        """이미지 크기 결정"""
        sizes = {
            "1:1": "1024x1024",
            "16:9": "1792x1024",
            "9:16": "1024x1792"
        }
        return sizes.get(aspect_ratio, "1024x1024")
    def create_scene_description(self, scene: str, config: SceneConfig) -> str:
    ###"""장면별 상세 시각적 설명 생성"""
        try:
            style_guide = self.style_guides[config.style]
            mood_guide = self.mood_guides[config.mood]
        
            prompt = f"""Webtoon Art Guidelines:
            Scene: {scene}
        
            Style Requirements:
            {style_guide['prompt']}
            {style_guide['emphasis']}
        
            Mood Requirements:
            {mood_guide['prompt']}
            Lighting: {mood_guide['lighting']}
            Color: {mood_guide['color']}
        
            Composition: {self.composition_guides[config.composition]}
            Character Traits: {config.character_desc if config.character_desc else '특별한 지정 없음'}
        
            Please provide detailed descriptions of the following elements:
1. Composition and perspective
2. Character positions, poses, and expressions
3. Depth and detail in the background
4. Handling of lighting and shadows
5. Visual elements that emphasize emotions"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
        
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Scene description creation failed: {str(e)}")
            raise


    def generate_image(self, description: str, config: SceneConfig) -> str:
        # 최대 시도 횟수 제한
        max_attempts = 3  
        min_acceptable_score = 0.6  # 최소 허용 점수

        for attempt in range(max_attempts):
            try:
                style_guide = self.style_guides[config.style]
                mood_guide = self.mood_guides[config.mood]
                
                final_prompt = f"""{description}
                Visual style: {style_guide['prompt']}
                Mood: {mood_guide['prompt']}
                Lighting: {mood_guide['lighting']}
                Color: {mood_guide['color']}"""

                # 부정적 프롬프트
                negative_prompt = """
                Abstract images, blurry images, low quality, unrealistic proportions,
                distorted faces, extra limbs, text within the image, speech bubbles, more than five characters, flags or countries,
                cropped images, excessive filters, ungrammatical structures, duplicated features,
                poor anatomy,overly complex backgrounds
                """


                # image_gen.py의 함수 사용
                image_url, revised_prompt, created_seed = generate_image_from_text(
                    prompt=final_prompt,
                    style=config.style,
                    aspect_ratio=config.aspect_ratio,
                    negative_prompt=negative_prompt
                )
                
                if image_url:
                    quality_check = self.clip_analyzer.validate_image(
                        image_url, 
                        description,
                        return_score=True
                    )
                    
                    score = quality_check.get("similarity_score", 0.0)
                    self._record_attempt(attempt, image_url, score)
                    
                    # 점수에 따른 조건부 수락
                    if score >= 0.7:  # target_score_threshold
                        logging.info(f"이상적인 이미지 생성 (점수: {score})")
                        return image_url
                    elif score >= min_acceptable_score and attempt >= 1:
                        logging.info(f"적정 수준의 이미지 생성 (점수: {score})")
                        return image_url
                    
                    # 프롬프트 개선은 1회만 시도
                    if attempt == 0 and score < min_acceptable_score:
                        description = self._enhance_prompt_with_missing_elements(
                            description,
                            quality_check.get("missing_elements", [])
                        )
                        logging.info("프롬프트 개선 시도")
                    
            except Exception as e:
                logging.error(f"이미지 생성 시도 {attempt + 1} 실패: {str(e)}")
                if attempt == max_attempts - 1:
                    best_result = self._get_best_attempt()
                    if best_result:
                        return best_result
                    
        return None

    def _record_attempt(self, attempt_num: int, image_url: str, score: float):
        """각 시도의 결과를 기록"""
        if not hasattr(self, '_generation_attempts'):
            self._generation_attempts = []
        
        self._generation_attempts.append({
            'attempt': attempt_num,
            'image_url': image_url,
            'score': score,
            'timestamp': datetime.now()
        })

    def _get_best_attempt(self) -> str:
        """지금까지의 시도 중 최상의 결과 반환"""
        if not hasattr(self, '_generation_attempts') or not self._generation_attempts:
            return None
            
        best_attempt = max(self._generation_attempts, key=lambda x: x['score'])
        logging.info(f"select best attempt (점수: {best_attempt['score']})")
        return best_attempt['image_url']

    def _enhance_prompt_with_missing_elements(self, original_prompt: str, missing_elements: list) -> str:
        """프롬프트 개선"""
        try:
            enhancement = f"""
            Required Elements to Include:
            - Character Traits: {', '.join(missing_elements) if missing_elements else 'Keep as is'}
            - Story Context: {original_prompt}

            Please integrate the above elements naturally and make them more detailed.
            Focus particularly on the character's actions and emotional expressions.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # 빠른 응답을 위해 GPT-3.5 사용
                messages=[
                    {"role": "system", "content": enhancement},
                    {"role": "user", "content": original_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            enhanced_prompt = response.choices[0].message.content.strip()
            logging.info("프롬프트 개선 완료")
            return enhanced_prompt
            
        except Exception as e:
            logging.error(f"프롬프트 개선 실패: {str(e)}")
            return original_prompt

    def summarize_scene(self, description: str, original_text: str = "", scene_index: int = 0) -> str:
        """
        동화책 삽화 스타일의 간결한 장면 설명 생성
        """
        try:
            # 프롬프트 구성
            prompt =f"""
            The following task involves writing a short description that will be placed below an illustration in a storybook.
            Please provide a concise summary of the current scene in English.

            Current Scene Description:
            "{description}"

            Original Text:
            "{original_text[:1000]}"  # The first 500 characters of the story are provided for context.

            Requirements:
            1. Summarize the core actions and dialogues of the current scene concisely.
            2. Remove unnecessary details and use natural sentences.
            3. The summary should not exceed 150 characters.
            4. Exclude irrelevant details and technical descriptions.
            5. Do not include the protagonist's emotions.

            Examples:
            - "The protagonist sits by the window, lost in thought, gazing at the distant scenery."
            - "Two people share an umbrella as they walk down a rainy street."
             ❌ "The character is walking in the vineyard, looking at grapes."
             ✅ "The fox smiles under the sunlight in the vineyard."
            """

        
            # GPT 모델 호출
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in crafting concise and natural summaries of webtoon scenes. "},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # 일관성 유지
                max_tokens=250  # 요약 길이 제한
            )
        
            # 응답 처리
            summary = response.choices[0].message.content.strip()
            if len(summary) > 250:  # 150자 제한
                summary = summary[:250].rsplit('.', 1)[0] + '.'
        
            return summary
    
        except Exception as e:
            logging.error(f"Scene summarization failed: {str(e)}")
            # 오류 시 기본 응답 반환
            return f"요약 실패: {description[:150]}..."


    def render_ui(self):
        st.title("Visualize Story Text")
         # UI 가이드 expander 추가
        with st.sidebar.expander("📌Interface Guide", expanded=True):
            st.markdown("""
       ### 🎨 Style Settings
        - **Minimalist**: Simple and clean design
        - **Pictogram**: Symbolic icon style
        - **Cartoon**: Exaggerated and dynamic expression
        - **Webtoon**: Korean comic style
        - **Artistic**: Creative and painterly expression
        
        ### 🌈 Mood Selection
        - **Casual**: Natural and relaxed tone
        - **Tense**: Dramatic and intense atmosphere
        - **Serious**: Weighty and formal expression
        - **Warm**: Cozy and positive emotions
        - **Joyful**: Bright and cheerful mood
        
        ### 📐 Composition Settings
        - **Background and Character**: Comprehensive scene composition
        - **Close-up Shot**: Emphasis on emotion and expressions
        - **Interactive**: Character interactions
        - **Landscape-focused**: Background-centered direction
        - **Default**: Basic composition
        """)
        
        st.info("""
    💡 **Tips for Use**
    - Choose a style that matches the mood of the story.
    - Select a mood that effectively conveys the emotions of the scene.
    - Set an appropriate composition for the situation to enhance the impact.
    """)

    
    # 세션 상태 초기화
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
    
        with st.form("story_input_form"):
            text_content = None
            if input_method == "Direct Input":
                text_content = st.text_area(
                    "Enter Story",
                    placeholder="Enter a novel, script, or any creative story.",
                    height=200
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload File",
                    type=['txt', 'pdf', 'docx', 'doc'],
                    help="Supported formats: TXT, PDF, DOCX"
                )
            
                if uploaded_file:
                    text_content = self.read_file_content(uploaded_file)
                    if text_content:
                        st.success("File uploaded successfully!")
                        with st.expander("View File Content"):
                            st.text(text_content[:500] + "..." if len(text_content) > 500 else text_content)
                
            col1, col2 = st.columns(2)
            with col1:
                style = st.select_slider(
                    "Select Style",
                    options=["Minimalist", "Pictogram", "Cartoon", "Webtoon", "Artistic"],
                    value="Webtoon"
                )
            
                mood = st.selectbox(
                    "Mood",
                    ["Casual", "Tense", "Serious", "Warm", "Joyful"]
                    )
            
                composition = st.selectbox(
                    "Composition",
                    ["Background and Character", "Close-up Shot", "Interactive", "Landscape-focused", "Default"]
                )
        
            with col2:
                character_desc = st.text_input(
                    "Character Description (Optional)",
                 placeholder="Enter the characteristics of the main character"
                )
            
                cut_count = st.radio(
                "Number of Cuts to Generate",
                options=[1, 2, 3, 4],
                horizontal=True
                )
            
                aspect_ratio = st.selectbox(
                "Image Aspect Ratio",
                ["Square (1:1)", "Wide (16:9)", "Portrait (9:16)"]
                )
        
            submit = st.form_submit_button("✨Start Webtoon Generation")
        
            if submit and text_content:
            # aspect ratio 값 변환
                ratio_map = {
                "Square (1:1)": "1:1",
                "Wide (16:9)": "16:9",
                "Portrait (9:16)": "9:16"
                }
            
                config = SceneConfig(
                    style=style,
                    composition=composition,
                    mood=mood,
                    character_desc=character_desc,
                    aspect_ratio=ratio_map.get(aspect_ratio, "1:1")
                )
            
                # 세션 상태에 현재 설정 저장
                st.session_state.current_config = config
                st.session_state.current_text = text_content
                self.process_submission(text_content, config, cut_count)

        # form 바깥에서 저장 버튼 처리
        if st.session_state.generated_images:
            if st.button("💾 Save this Session Prompts"):
                save_config = {
                    'type': 'story',
                    'title': st.session_state.current_text[:100],
                    'text': st.session_state.current_text,
                    'style': st.session_state.current_config.style,
                    'composition': st.session_state.current_config.composition,
                    'mood': st.session_state.current_config.mood,
                    'character_desc': st.session_state.current_config.character_desc,
                    'aspect_ratio': st.session_state.current_config.aspect_ratio,
                    'scene_descriptions': st.session_state.scene_descriptions
                }
                session_dir = save_session(save_config, st.session_state.generated_images)
                st.success(f"✅ Saved Successfully! location: {session_dir}")

    
    # process_submission 메소드 내의 이미지 생성 부분을 다음과 같이 수정

    def process_submission(self, text: str, config: SceneConfig, cut_count: int):
        try:
            progress_bar = st.progress(0)
            status = st.empty()
        
            # 로그 저장을 위한 세션 데이터 초기화
            if 'generation_logs' not in st.session_state:
                st.session_state.generation_logs = []
        
            # 분석 시작 시간 기록
            start_time = datetime.now()
        
            # CLIP 분석기 정보 표시
            st.sidebar.markdown("### 🔍CLIP Analysis Results")
            st.sidebar.info(f"Device: {self.clip_analyzer.device}")
            st.sidebar.info(f"Model: openai/clip-vit-base-patch32")
        
            status.info("📖 Analyzing story structure...")
            scenes = self.analyze_story_by_cuts(text, cut_count)
        
            generated_images = {}
            scene_descriptions = []
        
            # 생성 메트릭 저장용 딕셔너리
            generation_metrics = {
                'total_time': 0,
                'avg_clip_score': 0,
                'scores': [],
                'generation_attempts': []
            }
        
            cols_per_row = min(cut_count, 2)
            rows_needed = (cut_count + 1) // 2
        
            for row in range(rows_needed):
                cols = st.columns(cols_per_row)
                start_idx = row * cols_per_row
                end_idx = min(start_idx + cols_per_row, cut_count)
            
                for i in range(start_idx, end_idx):
                    scene_type, scene = list(scenes.items())[i]
                    status.info(f"🎨 {scene_type} Generating Scenes... ({i+1}/{cut_count})")
                
                    scene_start_time = datetime.now()
                
                    # 장면 설명 생성 및 CLIP 분석
                    description = self.create_scene_description(scene, config)
                    enhanced_description = self.clip_analyzer.enhance_prompt(
                        description, config.style, config.mood
                    )
                    scene_descriptions.append(enhanced_description)
                
                    # 이미지 생성
                    image_url = self.generate_image(enhanced_description, config)
                
                    if image_url:
                        generated_images[i] = image_url
                    
                        # CLIP 검증 및 품질 분석
                        quality_check = self.clip_analyzer.validate_image(
                            image_url, 
                            description,
                            return_score=True
                        )
                    
                        with cols[i % cols_per_row]:
                            # 이미지 표시
                            st.image(image_url, caption=f"cut {i+1}: {scene_type}", use_column_width=True)
                        
                            # 분석 결과 표시를 위한 expander 추가
                            with st.expander("🔍 CLIP Analyzing Result", expanded=False):
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
                            
                                # 세부 분석 결과 표시
                                st.write("Prompt Matching")
                                st.progress(score)
                            
                                # 생성 시간 표시
                                scene_time = (datetime.now() - scene_start_time).total_seconds()
                                st.info(f"⏱ Generation Time: {scene_time:.1f} seconds")
                        
                            # 장면 설명 표시
                            summary = self.summarize_scene(description, st.session_state.current_text, i)
                            st.markdown(
                                f"<p style='text-align: center; font-size: 14px;'>{summary}</p>",
                                unsafe_allow_html=True
                            )
                    
                        # 메트릭 업데이트
                        generation_metrics['scores'].append(score)
                        generation_metrics['generation_attempts'].append({
                            'scene_number': i + 1,
                            'scene_type': scene_type,
                            'clip_score': score,
                            'generation_time': scene_time
                        })
                
                    progress_bar.progress((i + 1) / cut_count)
        
            # 전체 생성 시간 계산
            generation_metrics['total_time'] = (datetime.now() - start_time).total_seconds()
            generation_metrics['avg_clip_score'] = sum(generation_metrics['scores']) / len(generation_metrics['scores'])
        
            # 생성 로그 저장
            st.session_state.generation_logs.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'config': config.__dict__,
                'metrics': generation_metrics
            })
        
            # 생성 결과 요약 표시
            st.sidebar.markdown("### 📊 Image Generation Results")
            st.sidebar.metric("Average CLIP Score", f"{generation_metrics['avg_clip_score']:.2f}")
            st.sidebar.metric("Total Generation Time", f"{generation_metrics['total_time']:.1f} seconds")
        
            # 세션 상태 업데이트
            st.session_state.generated_images = generated_images
            st.session_state.scene_descriptions = scene_descriptions
        
            status.success("✨ Webtoon Generation Complete!")
        
          
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            logging.error(f"Error in process_submission: {str(e)}")
def main():
    st.set_page_config(
        page_title="Text to Webtoon Converter",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        client = OpenAI()
        clip_analyzer = CLIPAnalyzer()
        converter = TextToWebtoonConverter(client, clip_analyzer)
        converter.render_ui()
    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()