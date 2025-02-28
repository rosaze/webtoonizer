import streamlit as st
from PIL import Image
import io
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv
from clip_analyzer import CLIPAnalyzer

# Import modules for each feature
from article_org import extract_news_info, simplify_terms_dynamically, generate_webtoon_scenes
from user_input import render_news_search, search_news, generate_final_prompt
from general_text_input import TextToWebtoonConverter
from nonfiction_input import NonFictionConverter

# Load .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.update({
        "page": "home",  # Default set to 'home'
        "selected_article": None,
        "article_content": None,
        "extracted_info": None,
        "simplified_content": None,
        "webtoon_episode": None,
        "current_cut_index": 0,
        "selected_images": {},
        "NAVER_CLIENT_ID": os.getenv("NAVER_CLIENT_ID"),
        "NAVER_CLIENT_SECRET": os.getenv("NAVER_CLIENT_SECRET")
    })

def render_home():
    st.title("Webtoonizer - Text Visualization Tool")
    
    # Layout columns for center alignment
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            padding: 20px;
        }
        .description {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .button-spacing {
            margin-bottom: 15px;
        }
        </style>
        <div class="big-font">Select your preferred visualization method</div>
        """, unsafe_allow_html=True)

        # Story text visualization button
        if st.button("📚 Visualize Story Text", use_container_width=True, key="story"):
            st.session_state.page = "text_input"
            st.rerun()

        st.markdown("""
        <div class="description">
        Convert novels, scripts, or stories into a webtoon format.
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")  # Add a separator line

        # Educational/scientific text visualization button
        if st.button("🎓 Visualize Educational/Scientific Text", use_container_width=True, key="edu"):
            st.session_state.page = "nonfiction_input"
            st.rerun()

        st.markdown("""
        <div class="description">
        Visually explain educational materials, scientific concepts, or processes.
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")  # Add a separator line

        # News visualization button (commented out in the original)
        # if st.button("📰 Visualize News", use_container_width=True, key="news"):
        #     st.session_state.page = "news_search"
        #     st.rerun()

        # st.markdown("""
        # <div class="description">
        # Search for the latest news articles and convert them into webtoon format.
        # </div>
        # """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Webtoonizer",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add a "Go to Home" button in the sidebar
    with st.sidebar:
        if st.button("🏠 Go to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        
        st.markdown("---")

    # Page routing
    if st.session_state.page == "home":
        render_home()
        
    elif st.session_state.page == "text_input":
        try:
            clip_analyzer = CLIPAnalyzer()
            converter = TextToWebtoonConverter(client, clip_analyzer)
            converter.render_ui()
        except Exception as e:
            st.error(f"Error processing text input: {str(e)}")
            
    elif st.session_state.page == "nonfiction_input":
        try:
            converter = NonFictionConverter(client)
            converter.render_ui()
        except Exception as e:
            st.error(f"Error processing educational/scientific content: {str(e)}")
  
    # Handle errors
    try:
        if st.session_state.get("error"):
            st.error(st.session_state.error)
            st.session_state.error = None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
