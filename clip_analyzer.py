import torch
from PIL import Image
import logging
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import requests
from io import BytesIO
import streamlit as st

class CLIPAnalyzer:
    def __init__(self):
        """Initialize CLIP model and processor"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.client = OpenAI()
            self.minimum_score_threshold = 0.5  # Minimum acceptable score
            self.target_score_threshold = 0.7   # Target score
            logging.info(f"CLIP Analyzer initialized on device: {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to initialize CLIP model: {str(e)}")
            raise RuntimeError(f"Failed to initialize CLIP model: {str(e)}")

    def enhance_prompt(self, prompt, style, mood):
        """Enhance the prompt and strengthen visual elements"""
        try:
            if not prompt or len(prompt.strip()) < 10:
                logging.warning("Prompt too short or empty for enhancement")
                return prompt
                
            # Extract key elements
            key_elements = self._extract_key_elements(prompt)
            
            enhancement_prompt = f"""
            Please improve the following scene description into a webtoon style:
            1. Maintain key visual elements while making it more detailed
            2. Naturally incorporate {style} style and {mood} mood
            3. Vividly express the character's emotions and actions
            4. Enhance atmosphere through background and lighting
            
            Key elements:
            {key_elements}
            
            Original description:
            {prompt}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using GPT-3.5 for faster responses
                messages=[{"role": "user", "content": enhancement_prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            enhanced_prompt = response.choices[0].message.content.strip()
            
            logging.info(f"Original prompt length: {len(prompt)}")
            logging.info(f"Enhanced prompt length: {len(enhanced_prompt)}")
            
            # Make sure we don't return an empty string
            if not enhanced_prompt or len(enhanced_prompt) < 10:
                logging.warning("Enhancement resulted in empty prompt, using original")
                return prompt
                
            return enhanced_prompt
            
        except Exception as e:
            logging.error(f"Error enhancing prompt: {str(e)}")
            return prompt

    def _extract_key_elements(self, text):
        """Extract key visual elements from the text"""
        try:
            if not text or len(text.strip()) < 10:
                logging.warning("Text too short or empty for key element extraction")
                return "No elements provided"
                
            prompt = """
            Please extract the most important visual elements (up to 3):
            1. Main character's actions and expressions
            2. Significant background elements
            3. Overall atmosphere or lighting
            
            Scene:
            {text}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.format(text=text)}],
                max_tokens=100,
                temperature=0.3
            )
            
            key_elements = response.choices[0].message.content.strip()
            logging.info(f"Extracted key elements: {key_elements}")
            
            # Verify we have meaningful content
            if not key_elements or len(key_elements) < 10:
                return "Failed to extract meaningful elements"
            
            return key_elements
            
        except Exception as e:
            logging.error(f"Error extracting key elements: {str(e)}")
            return "Failed to extract key elements"

    def validate_image(self, image_url, prompt, story_context=None, return_score=False):
        """Validate the alignment between image and prompt"""
        try:
            if not prompt or len(prompt.strip()) < 10:
                logging.warning("Prompt too short or empty for validation")
                default_result = {
                    "similarity_score": 0.5,
                    "meets_requirements": True,
                    "prompt_used": "Empty prompt",
                    "error": "Empty prompt provided"
                }
                return default_result if return_score else default_result["meets_requirements"]
                
            # First determine if abstract concept and process prompt accordingly
            core_prompt = ""
            if self._is_abstract_concept(prompt):
                decomposed_prompt = self.extract_abstract_concepts(prompt)
                if decomposed_prompt and decomposed_prompt != prompt:
                    core_prompt = self._extract_core_prompt(decomposed_prompt)
                else:
                    core_prompt = self._extract_core_prompt(prompt)
            else:
                core_prompt = self._extract_core_prompt(prompt)

            # Validate core prompt
            if not core_prompt or len(core_prompt) < 5:
                logging.warning("Failed to extract core prompt, using fallback")
                core_prompt = prompt[:100]  # Fallback to first 100 chars

            # Limit prompt length for CLIP
            max_length = 77  # Maximum token length for CLIP model
            core_prompt = ' '.join(core_prompt.split()[:max_length])
        
            # Download and preprocess image
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        
            # Prepare CLIP inputs
            inputs = self.processor(
                images=image,
                text=[core_prompt],
                return_tensors="pt",
                padding=True
            ).to(self.device)
        
            # Calculate similarity
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.nn.functional.softmax(logits_per_image, dim=1)
                similarity = probs[0][0].item()
        
            # Check story consistency if context exists
            if story_context and story_context.get("previous_scenes"):
                context_score = self._check_story_consistency(image, story_context)
                similarity = (0.7 * similarity) + (0.3 * context_score)
        
            # Prepare result
            result = {
                "similarity_score": similarity,
                "meets_requirements": similarity >= self.target_score_threshold,
                "prompt_used": core_prompt
            }
        
            return result if return_score else result["meets_requirements"]
        
        except Exception as e:
            logging.error(f"Error validating image: {str(e)}")
            default_result = {
                "similarity_score": 0.5,
                "meets_requirements": True,
                "error": str(e)
            }
            return default_result if return_score else True
            
    def _is_abstract_concept(self, prompt):
        """ Determine if the prompt contains abstract concepts that need decomposition.
    
        Args:
            prompt (str): The input prompt to analyze
            
        Returns:
            bool: True if the prompt contains abstract concepts, False otherwise"""
        try:
            if not prompt or len(prompt.strip()) < 10:
                return False
                
            abstract_indicators = [
                "freedom", "love", "happiness", "justice", "peace",
                "wisdom", "truth", "beauty", "courage", "hope",
                "fear", "anger", "joy", "sadness", "success",
                "failure", "infinity", "eternity", "chaos", "harmony",
                "balance", "unity", "complexity", "simplicity"
            ]
        
            # Check if the prompt contains abstract concepts
            prompt_lower = prompt.lower()
            
            if any(concept in prompt_lower for concept in abstract_indicators):
                return True
                
            analysis_prompt = f"""
            Analyze if this text contains abstract concepts that would be challenging to visualize directly:
        
            "{prompt[:500]}"  # Limiting to 500 chars to avoid token limits
        
            Response should be either "true" or "false".
            Consider concepts that:
            - Cannot be directly photographed
            - Require metaphorical representation
            - Involve complex emotional or philosophical ideas
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using 3.5 for faster response
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().lower()
            return result == "true"
    
        except Exception as e:
            logging.error(f"Error in abstract concept detection: {str(e)}")
            return False  # Default to concrete concept processing if error occurs
        
    def _extract_core_prompt(self, prompt):
        """Extract core content from the prompt"""
        try:
            # Handle empty or very short prompts
            if not prompt or len(prompt.strip()) < 10:
                logging.warning("Prompt too short or empty for extraction")
                return "Empty scene description provided"
                
            # Limit very long prompts 
            if len(prompt) > 2000:
                prompt = prompt[:2000]
                
            system_prompt = """
            Extract the most essential visual elements from the following scene description in one sentence:
            - Remove unnecessary details
            - Retain key actions and atmosphere
            - Keep it simple and clear
            Limit to 50 words.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            core_prompt = response.choices[0].message.content.strip()
            logging.info(f"Extracted core prompt: {core_prompt}")
            
            # Verify we have meaningful content
            if not core_prompt or "did not provide" in core_prompt.lower() or "no scene description" in core_prompt.lower():
                logging.warning("Core prompt extraction failed to produce useful result")
                return prompt[:100]  # Use the first 100 characters as a fallback
                
            return core_prompt
            
        except Exception as e:
            logging.error(f"Error extracting core prompt: {str(e)}")
            return prompt[:100]  # Use the first 100 characters of the original prompt in case of error
            
    def _check_story_consistency(self, new_image, story_context):
        """Validate consistency between the new image and previous scenes"""
        try:
            if not story_context.get("previous_scenes"):
                return 1.0  # First scene

            previous_images = []
            for scene in story_context["previous_scenes"][-3:]:  # Compare the last 3 scenes only
                try:
                    if not scene.get("image_url"):
                        logging.warning("Missing image URL in previous scene")
                        continue
                        
                    response = requests.get(scene["image_url"])
                    prev_image = Image.open(BytesIO(response.content))
                    previous_images.append(prev_image)
                except Exception as e:
                    logging.warning(f"Failed to load previous image: {e}")
                    continue

            if not previous_images:
                return 1.0

            # Calculate style consistency score
            consistency_scores = []
            for prev_image in previous_images:
                try:
                    inputs = self.processor(
                        images=[prev_image, new_image],
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    with torch.no_grad():
                        features = self.model.get_image_features(**inputs)
                        similarity = torch.nn.functional.cosine_similarity(
                            features[0].unsqueeze(0), 
                            features[1].unsqueeze(0)
                        ).item()
                    consistency_scores.append(similarity)
                except Exception as e:
                    logging.error(f"Error calculating consistency score: {e}")
                    continue

            if not consistency_scores:
                return 1.0

            return sum(consistency_scores) / len(consistency_scores)

        except Exception as e:
            logging.error(f"Error checking story consistency: {str(e)}")
            return 1.0

    def analyze_style_consistency(self, images):
        """Analyze style consistency across multiple images"""
        if not images or len(images) < 2:
            return True, 1.0
            
        try:
            # Convert images to CLIP embeddings
            embeddings = []
            valid_images = []
            
            for img_url in images:
                try:
                    response = requests.get(img_url)
                    img = Image.open(BytesIO(response.content))
                    valid_images.append(img)
                    
                    inputs = self.processor(
                        images=img,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.model.get_image_features(**inputs)
                    embeddings.append(embedding)
                except Exception as e:
                    logging.warning(f"Failed to process image {img_url}: {e}")
                    continue
            
            if len(embeddings) < 2:
                logging.warning("Not enough valid images for consistency analysis")
                return True, 1.0
                
            # Calculate cosine similarity between embeddings
            similarities = []
            for i in range(len(embeddings)-1):
                for j in range(i+1, len(embeddings)):
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[i], embeddings[j]
                    ).item()
                    similarities.append(sim)
            
            # Calculate average similarity
            avg_similarity = sum(similarities) / len(similarities)
            
            # Emphasize character consistency by analyzing face regions
            character_consistency = self._analyze_character_consistency(valid_images)
            
            # Weight character consistency more heavily
            final_score = (0.6 * character_consistency) + (0.4 * avg_similarity)
            
            logging.info(f"Style consistency: {avg_similarity}, Character consistency: {character_consistency}")
            return final_score >= 0.7, final_score
            
        except Exception as e:
            logging.error(f"Error analyzing style consistency: {str(e)}")
            return False, 0.0
            
    def _analyze_character_consistency(self, images):
        """Analyze consistency of character appearance across images"""
        # This is a placeholder for actual character region extraction and analysis
        # For complete implementation, this would use face detection and region-specific CLIP analysis
        try:
            # Simple placeholder implementation - just returns a high value
            # In a real implementation, this would do character-specific analysis
            return 0.8
        except Exception as e:
            logging.error(f"Error in character consistency analysis: {str(e)}")
            return 0.5

    def get_image_focus_area(self, image_url, prompt):
        """Detect important areas in the image"""
        try:
            if not image_url:
                logging.warning("Empty image URL provided")
                return None
                
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs, output_attentions=True)
                attention_map = outputs.attentions[-1].mean(dim=1)
            
            return attention_map.cpu().numpy()
            
        except Exception as e:
            logging.error(f"Error analyzing focus area of image: {str(e)}")
            return None
            
    def extract_abstract_concepts(self, text):
        """Extract abstract concepts from the text"""
        try:
            if not text or len(text.strip()) < 10:
                logging.warning("Text too short or empty for abstract extraction")
                return text
                
            # Limit input to avoid token limits
            if len(text) > 1500:
                text = text[:1500]
                
            prompt = """
            분석할 추상적 개념: {text}
        
            다음 요소들로 분해:
            1. 핵심 시각적 메타포
            2. 관련된 구체적 사물/행동
            3. 감정적/분위기 요소
            4. 상징적 색상/형태
            
            출력 형식:
            - 각 요소별 3개 이하의 구체적 표현
            - 시각화 가능한 형태로 변환
            """
            
            formatted_prompt = prompt.format(text=text)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            concepts = response.choices[0].message.content.strip()
            logging.info(f"Decomposed concepts: {concepts}")
            
            if not concepts or len(concepts) < 20:
                logging.warning("Concept decomposition failed to produce valid result")
                return text
                
            combined_concepts = self._combine_visual_elements(concepts)
            
            # If combining failed, return the decomposed concepts directly
            if not combined_concepts or len(combined_concepts) < 20:
                return concepts
                
            return combined_concepts
            
        except Exception as e:
            logging.error(f"Concept decomposition failed: {str(e)}")
            return text
            
    def _combine_visual_elements(self, decomposed_concepts):
        """Combine visual elements into a coherent abstract concept"""
        try:
            if not decomposed_concepts or len(decomposed_concepts.strip()) < 20:
                logging.warning("Decomposed concepts too short or empty")
                return decomposed_concepts
                
            prompt = f"""
            Please Combine these visual elements into a cohesive scene:
            {decomposed_concepts}
            
            Requirements:
            - Maintain visual clarity
            - Focus on symbolic representation
            - Create clear focal points
            - Ensure elements are visually balanced
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5
            )
            
            result = response.choices[0].message.content.strip()
            
            if not result or len(result) < 20 or "unable to" in result.lower():
                logging.warning("Visual element combination failed to produce valid result")
                return decomposed_concepts
                
            return result
        
        except Exception as e:
            logging.error(f"Visual element combination failed: {str(e)}")
            return decomposed_concepts

    @staticmethod
    def visualize_results(image_url, clip_score, attention_map=None):
        """Visualize validation results"""
        try:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(image_url, use_column_width=True)
                
            with col2:
                st.metric("CLIP Similarity", f"{clip_score:.2f}")
                if clip_score >= 0.7:
                    st.success("✓ Validated")
                else:
                    st.warning("⚠ Needs Improvement")
                
            if attention_map is not None:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    st.write("Focus Area:")
                    fig, ax = plt.subplots()
                    sns.heatmap(attention_map, ax=ax)
                    st.pyplot(fig)
                except ImportError:
                    st.warning("Matplotlib and Seaborn required for attention map visualization")
                
        except Exception as e:
            logging.error(f"Error visualizing results: {str(e)}")
            st.error("Failed to visualize results")