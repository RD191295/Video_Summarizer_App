import streamlit as st
from phi.agent import Agent 
from phi.model.google import Gemini 
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai 
import time
from pathlib import Path
import tempfile
import os

# Page Configuration 
global gemini_api_key

st.set_page_config(
    page_title= "Multimodel AI Agent - Video Summarizer",
    page_icon= "🎥",
    layout= 'wide',
    initial_sidebar_state="expanded",  # Expand or collapse the sidebar
)

# Custom CSS to style the title
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        font-family: 'Arial', sans-serif;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">PhiData Video AI Summarizer Agent 🤖</div>', unsafe_allow_html=True)
st.header("Powerd by Gemini 2.0 Flash Exp")

with st.sidebar:
    st.title("API Key")
    gemini_api_key = st.text_input("Gemini API Key", key="file_qa_api_key", type="password")

    "[View the source code](https://github.com/RD191295/Video_Summarizer_App)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RD191295/Video_Summarizer_App?quickstart=1)"

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

@st.cache_resource
def initialize_agent():
    return Agent(
        name = "Video AI Summarizer",
        model = Gemini(id = 'gemini-2.0-flash-exp', api_key= gemini_api_key),
        tools = [DuckDuckGo()],
        markdown= True
    )

## Initialize the Agent

multimodel_Agent = initialize_agent()

# video file uploader
video_file = st.file_uploader(
    "Upload a Video File", type = ['mp4','mov','avi'] , help="Upload a Video for AI Analysis"
)

if video_file :
    with tempfile.NamedTemporaryFile(delete= False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format= 'video/mp4', start_time= 0)

    user_query = st.text_area(
        "What insights are you seeking from the video",
        placeholder='Ask Anything about video content. AI Agent will analyze and gather addition information from Search to answer query',
        help = 'Provide specific questions or insights you want from video'
    )

    st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #ff4b4b; /* Red background */
        color: white; /* White text */
        font-size: 16px; /* Optional: Increase font size */
        font-weight: bold; /* Optional: Make text bold */
        border: 2px solid #ff4b4b; /* Border matches background color */
        border-radius: 10px; /* Rounded corners */
        padding: 10px 20px;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff6666; /* Lighter red on hover */
        color: white; /* Ensure text stays white on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    if st.button("🔍 Analyze Video", key = 'analyze_video_button'):
        if not user_query:
            st.warning("Please enter a question or insights to analyze the video")
        
        else:
            try:
                with st.spinner("Processing video and gathering insights...."):
                    # upload and process file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                    
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context
                        Resopnd to the following query using cvideo insights and supplementry web search if required.
                        user Query : {user_query}

                        Make sure your analysis is detailed, user-friendly and actionable response.
                        """
                    )

                    # AI Agent Processing
                    response = multimodel_Agent.run(analysis_prompt,videos=[processed_video])

                    st.markdown(response.content)

                # Display the result
                st.subheader("Analysis result")
                st.markdown(response.content)
            except Exception as error:
                st.error(f'An error occured during analysis : {error}')
            finally:
                # clean up temp file
                Path(video_path).unlink(missing_ok=True)

else:
    st.info("Please Upload a video file to begin analysis")

# Customize text are height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Footer
footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 80px;
            background-color: #343a40;
            color: #ffffff;
            text-align: center;
            padding: 14px 0;
            font-size: 16px;
            font-family: 'Arial', sans-serif;
            border-top: 1px solid #444;
        }
        .footer:hover{
            color: #f1c40f;  /* Light link color */
            text-decoration: underline;
        }
    </style>
    <div class="footer">
        <p>&copy; 2025 My Streamlit App | <a href="https://www.example.com" target="_blank">Website</a> | <a href="https://www.linkedin.com/in/raj-dalsaniya/" target="_blank">LinkedIn</a></p>
        <p>All Rights Reserved</p>

    </div>
"""

# Display the footer
st.markdown(footer, unsafe_allow_html=True)

