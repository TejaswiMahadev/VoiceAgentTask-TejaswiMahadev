import streamlit as st
import speech_recognition as sr
import requests
import io
import base64
from pathlib import Path
import tempfile
import os
from typing import Optional, List, Dict, Any
import time
import json
from dataclasses import dataclass
from datetime import datetime
import warnings
import hashlib

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Audio processing - Fixed imports
try:
    import numpy as np
    from audio_recorder_streamlit import audio_recorder
    import wave
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError as e:
    st.error(f"Audio libraries not available: {e}")
    AUDIO_AVAILABLE = False

# Set user agent
os.environ['USER_AGENT'] = 'VoiceAssistant/1.0'

@dataclass
class VoiceAssistantConfig:
    """Configuration for the voice assistant"""
    gemini_api_key: str
    elevenlabs_api_key: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    model_name: str = "gemini-1.5-flash"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 500
    temperature: float = 0.7

class SimpleMemory:
    """Simple conversation memory"""
    
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, human_message: str, ai_message: str):
        self.messages.append(("human", human_message))
        self.messages.append(("ai", ai_message))
        
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        history = []
        for role, message in self.messages:
            if role == "human":
                history.append(f"User: {message}")
            else:
                history.append(f"Assistant: {message}")
        return "\n".join(history[-10:])
    
    def clear(self):
        self.messages = []

class DocumentProcessor:
    """Handles document loading and processing for RAG"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            loader = WebBaseLoader(url, header_template=headers)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"Error loading URL: {str(e)}")
            return []
    
    def load_from_pdf(self, pdf_file) -> List[Document]:
        """Load documents from a PDF file"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            os.unlink(tmp_file_path)
            
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return []
    
    def create_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        """Create FAISS vector store from documents"""
        if not documents:
            return None
        
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

class SpeechProcessor:
    """Handles speech-to-text and text-to-speech - FIXED VERSION"""
    
    def __init__(self, elevenlabs_api_key: str, voice_id: str):
        self.elevenlabs_api_key = elevenlabs_api_key
        self.voice_id = voice_id
        self.recognizer = sr.Recognizer()
        
        # Improved recognizer settings
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        
        # Check for alternative TTS options
        self.tts_engine = None
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            # Configure pyttsx3
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            self.tts_engine.setProperty('rate', 180)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume
        except ImportError:
            self.tts_engine = None
    
    def speech_to_text(self, audio_data) -> Optional[str]:
        """Convert speech to text - IMPROVED VERSION"""
        if not audio_data:
            return None
            
        try:
            # Create a temporary WAV file with proper format
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            # Debug: Check file size
            file_size = os.path.getsize(tmp_file_path)
            if file_size < 1000:  # Less than 1KB likely means no audio
                st.warning("Audio file seems too small. Please try recording again.")
                os.unlink(tmp_file_path)
                return None
            
            # Use speech recognition with better error handling
            with sr.AudioFile(tmp_file_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
                
                # Try multiple recognition services as fallback
                try:
                    text = self.recognizer.recognize_google(audio, language='en-US')
                    st.success(f"🎤 Recognized: '{text}'")
                except sr.UnknownValueError:
                    try:
                        # Fallback to whisper if available
                        text = self.recognizer.recognize_whisper(audio)
                        st.success(f"🎤 Recognized (Whisper): '{text}'")
                    except:
                        st.error("Could not understand the audio. Please speak clearly and try again.")
                        return None
                except sr.RequestError as e:
                    st.error(f"Speech recognition service error: {e}")
                    return None
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            return text
            
        except Exception as e:
            st.error(f"Error processing speech: {str(e)}")
            if 'tmp_file_path' in locals():
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            return None
    
    def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech - IMPROVED WITH FALLBACKS"""
        if not text or not text.strip():
            return None
        
        # Clean and limit text
        text = text.strip()
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        # Remove problematic characters
        text = text.replace('"', "'").replace('\n', '. ')
        
        # Try ElevenLabs first if API key is provided
        if self.elevenlabs_api_key:
            try:
                return self._elevenlabs_tts(text)
            except Exception as e:
                st.warning(f"ElevenLabs failed: {str(e)}. Trying fallback...")
        
        # Fallback to local TTS
        if self.tts_engine:
            try:
                return self._local_tts(text)
            except Exception as e:
                st.error(f"Local TTS failed: {str(e)}")
        
        # Fallback to Google TTS
        try:
            return self._google_tts(text)
        except Exception as e:
            st.error(f"All TTS methods failed. Last error: {str(e)}")
            return None
    
    def _elevenlabs_tts(self, text: str) -> Optional[bytes]:
        """ElevenLabs TTS with better error handling"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_api_key
        }
        
        # Test API key first
        test_url = "https://api.elevenlabs.io/v1/user"
        test_response = requests.get(test_url, headers={"xi-api-key": self.elevenlabs_api_key}, timeout=10)
        
        if test_response.status_code != 200:
            raise Exception(f"Invalid API key or account issue (Status: {test_response.status_code})")
        
        # Improved voice settings
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.6,
                "similarity_boost": 0.8,
                "style": 0.2,
                "use_speaker_boost": True
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        if response.status_code == 200:
            audio_content = response.content
            if len(audio_content) > 1000:
                st.success("🔊 ElevenLabs TTS successful")
                return audio_content
            else:
                raise Exception("Generated audio is too short")
        elif response.status_code == 401:
            raise Exception("Invalid API key")
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded")
        elif response.status_code == 422:
            raise Exception("Text content issue")
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    
    def _local_tts(self, text: str) -> Optional[bytes]:
        """Local TTS using pyttsx3"""
        import tempfile
        import wave
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Save to file
        self.tts_engine.save_to_file(text, tmp_path)
        self.tts_engine.runAndWait()
        
        # Read the file back
        with open(tmp_path, 'rb') as f:
            audio_data = f.read()
        
        # Clean up
        os.unlink(tmp_path)
        
        if len(audio_data) > 1000:
            st.success("🔊 Local TTS successful")
            return audio_data
        else:
            raise Exception("Local TTS generated empty audio")
    
    def _google_tts(self, text: str) -> Optional[bytes]:
        """Google TTS using gTTS as fallback"""
        try:
            from gtts import gTTS
            import tempfile
            
            tts = gTTS(text=text, lang='en', slow=False)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tts.save(tmp_path)
            
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            os.unlink(tmp_path)
            
            if len(audio_data) > 1000:
                st.success("🔊 Google TTS successful")
                return audio_data
            else:
                raise Exception("Google TTS generated empty audio")
                
        except ImportError:
            raise Exception("gTTS not available. Install with: pip install gtts")

class VoiceAssistant:
    """Main voice assistant class"""
    
    def __init__(self, config: VoiceAssistantConfig):
        self.config = config
        self.memory = SimpleMemory()
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=config.model_name,
                google_api_key=config.gemini_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.gemini_api_key
            )
            
            self.doc_processor = DocumentProcessor(self.embeddings)
            self.speech_processor = SpeechProcessor(
                config.elevenlabs_api_key, 
                config.voice_id
            )
            
            self.vector_store = None
            self.system_prompt = ""
            self.initialization_error = None
            
        except Exception as e:
            self.initialization_error = str(e)
            st.error(f"Error initializing assistant: {str(e)}")
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the assistant"""
        self.system_prompt = prompt
    
    def load_knowledge_base(self, documents: List[Document]):
        """Load documents into the knowledge base"""
        if self.initialization_error:
            st.error("Cannot load knowledge base due to initialization error")
            return
            
        try:
            self.vector_store = self.doc_processor.create_vector_store(documents)
            if self.vector_store:
                st.success("Knowledge base loaded successfully")
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")
    
    def get_response(self, question: str) -> str:
        """Get response from the assistant"""
        if self.initialization_error:
            return "I'm sorry, there was an error initializing the assistant. Please check your API keys and try again."
        
        try:
            context = ""
            if self.vector_store:
                retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
                docs = retriever.get_relevant_documents(question)
                context = "\n".join([doc.page_content for doc in docs])
            
            history = self.memory.get_conversation_history()
            
            full_prompt = f"""System Instructions:
{self.system_prompt}

Context from knowledge base:
{context}

Conversation History:
{history}

User: {question}

Please provide a helpful, natural response based on the context and conversation history. Keep responses concise but informative (under 200 words for voice responses)."""

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.llm.invoke(full_prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    self.memory.add_message(question, response_text)
                    return response_text
                    
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            st.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            return "I'm experiencing high demand right now. Please try again in a few moments."
                    else:
                        st.error(f"Error generating response: {str(e)}")
                        return "I apologize, but I'm having trouble processing your request right now."
            
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now."
    
    def process_voice_input(self, audio_data) -> Optional[str]:
        """Process voice input and return text"""
        return self.speech_processor.speech_to_text(audio_data)
    
    def generate_voice_response(self, text: str) -> Optional[bytes]:
        """Generate voice response from text"""
        return self.speech_processor.text_to_speech(text)

def main():
    st.set_page_config(
        page_title="AI Voice Assistant",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 AI Voice Assistant")
    st.markdown("A conversational AI assistant with voice capabilities")
    
    # Check audio availability and TTS options
    if not AUDIO_AVAILABLE:
        st.error("⚠️ Audio libraries not available. Please install: pip install audio-recorder-streamlit pyaudio wave")
    
    # TTS availability check
    tts_options = []
    try:
        import pyttsx3
        tts_options.append("Local TTS (pyttsx3)")
    except ImportError:
        pass
    
    try:
        import gtts
        tts_options.append("Google TTS (gTTS)")
    except ImportError:
        pass
    
    if tts_options:
        st.success(f"🔊 Available TTS: {', '.join(tts_options)}")
    else:
        st.warning("⚠️ No fallback TTS available. Install with: pip install pyttsx3 gtts")
    
    # API quota warning
    st.info("💡 **Tip**: If you encounter rate limit errors, try using Gemini Flash model or upgrade your API plan.")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Keys
    gemini_api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get your API key from Google AI Studio"
    )
    
    # ElevenLabs section with troubleshooting
    st.sidebar.subheader("🔊 Voice Settings")
    elevenlabs_api_key = st.sidebar.text_input(
        "ElevenLabs API Key (Optional)",
        type="password",
        help="Get your API key from ElevenLabs - leave empty to use fallback TTS"
    )
    
    if elevenlabs_api_key:
        voice_id = st.sidebar.text_input(
            "ElevenLabs Voice ID",
            value="21m00Tcm4TlvDq8ikWAM",
            help="Voice ID from ElevenLabs Voice Lab"
        )
        
        # Test ElevenLabs connection
        if st.sidebar.button("🧪 Test ElevenLabs API"):
            try:
                test_url = "https://api.elevenlabs.io/v1/user"
                test_response = requests.get(
                    test_url, 
                    headers={"xi-api-key": elevenlabs_api_key}, 
                    timeout=10
                )
                if test_response.status_code == 200:
                    user_data = test_response.json()
                    st.sidebar.success(f"✅ API Key Valid! Credits: {user_data.get('subscription', {}).get('character_count', 'Unknown')}")
                else:
                    st.sidebar.error(f"❌ API Key Invalid (Status: {test_response.status_code})")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
    else:
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        st.sidebar.info("💡 Using fallback TTS options (Local/Google)")
    
    model_choice = st.sidebar.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"],
        help="Flash model has higher rate limits"
    )
    
    # System Prompt
    st.sidebar.header("🤖 Assistant Configuration")
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        value="""Your name is Nira. You are the AI assistant for Analytas — an AI advisory firm that helps organizations deploy autonomous AI agents safely and intelligently. Your primary goal is to assist website visitors exploring whether AI agents are right for their use case. Begin by asking what brought them to Analytas today. Answer questions clearly, calmly, and with nuance. Your tone should be professional, trustworthy, and conversational.""",
        height=150
    )
    
    # Knowledge Base
    st.sidebar.header("📚 Knowledge Base")
    knowledge_source = st.sidebar.selectbox(
        "Knowledge Source",
        ["None", "Website URL", "PDF Upload"]
    )
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'last_audio_hash' not in st.session_state:
        st.session_state.last_audio_hash = None
    if 'processing_audio' not in st.session_state:
        st.session_state.processing_audio = False
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Conversation")
        
        # Check required API keys
        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar to continue.")
            st.markdown("Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
            return
        
        # Initialize assistant
        if st.session_state.assistant is None:
            config = VoiceAssistantConfig(
                gemini_api_key=gemini_api_key,
                elevenlabs_api_key=elevenlabs_api_key or "",
                voice_id=voice_id,
                model_name=model_choice
            )
            with st.spinner("🔄 Initializing assistant..."):
                st.session_state.assistant = VoiceAssistant(config)
                st.session_state.assistant.set_system_prompt(system_prompt)
        
        # Load knowledge base
        if knowledge_source != "None" and not st.session_state.documents_loaded:
            documents = []
            
            if knowledge_source == "Website URL":
                url = st.sidebar.text_input("Website URL", placeholder="https://example.com")
                if url and st.sidebar.button("Load Website"):
                    with st.spinner("🌐 Loading website content..."):
                        documents = st.session_state.assistant.doc_processor.load_from_url(url)
                        if documents:
                            st.session_state.assistant.load_knowledge_base(documents)
                            st.session_state.documents_loaded = True
                            st.success(f"✅ Loaded {len(documents)} document chunks")
            
            elif knowledge_source == "PDF Upload":
                uploaded_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])
                if uploaded_file and st.sidebar.button("Load PDF"):
                    with st.spinner("📄 Loading PDF content..."):
                        documents = st.session_state.assistant.doc_processor.load_from_pdf(uploaded_file)
                        if documents:
                            st.session_state.assistant.load_knowledge_base(documents)
                            st.session_state.documents_loaded = True
                            st.success(f"✅ Loaded {len(documents)} document chunks")
        
        # Display conversation
        chat_container = st.container()
        with chat_container:
            for i, (role, message, audio) in enumerate(st.session_state.conversation):
                if role == "user":
                    st.chat_message("user").write(message)
                else:
                    with st.chat_message("assistant"):
                        st.write(message)
                        if audio and elevenlabs_api_key:
                            st.audio(audio, format='audio/mpeg')
        
        # Voice input section - IMPROVED
        if AUDIO_AVAILABLE:
            st.subheader("🎤 Voice Input")
            
            # Show TTS status
            tts_status = []
            if elevenlabs_api_key:
                tts_status.append("ElevenLabs")
            try:
                import pyttsx3
                tts_status.append("Local")
            except ImportError:
                pass
            try:
                import gtts
                tts_status.append("Google")
            except ImportError:
                pass
            
            if tts_status:
                st.info(f"🔊 TTS Options: {' → '.join(tts_status)}")
            else:
                st.warning("⚠️ No TTS available")
            
            col_voice1, col_voice2 = st.columns([3, 1])
            
            with col_voice1:
                audio_bytes = audio_recorder(
                    text="🎤 Click to record your voice",
                    recording_color="#ff6b6b",
                    neutral_color="#4ecdc4",
                    icon_name="microphone",
                    icon_size="2x",
                    key="voice_recorder"
                )
            
            with col_voice2:
                if st.button("🗑️ Clear Audio", help="Clear the current audio recording"):
                    st.session_state.last_audio_hash = None
                    st.rerun()
            
            # Process audio with better state management
            if audio_bytes and not st.session_state.processing_audio:
                # Create hash to prevent reprocessing
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                
                if audio_hash != st.session_state.last_audio_hash:
                    st.session_state.last_audio_hash = audio_hash
                    st.session_state.processing_audio = True
                    
                    with st.spinner("🔍 Processing voice input..."):
                        text_input = st.session_state.assistant.process_voice_input(audio_bytes)
                        
                        if text_input and text_input.strip():
                            # Process the voice input
                            with st.spinner("🤔 Generating response..."):
                                response = st.session_state.assistant.get_response(text_input)
                                
                                # Generate voice response
                                voice_response = None
                                if tts_status:  # If any TTS option is available
                                    with st.spinner("🔊 Generating voice response..."):
                                        voice_response = st.session_state.assistant.generate_voice_response(response)
                                
                                # Add to conversation
                                st.session_state.conversation.append(("user", text_input, None))
                                st.session_state.conversation.append(("assistant", response, voice_response))
                    
                    st.session_state.processing_audio = False
                    st.rerun()
        elif not AUDIO_AVAILABLE:
            st.info("💡 Install audio libraries to enable voice input: pip install audio-recorder-streamlit pyaudio wave")
        else:
            st.info("💡 Voice input available, but no TTS configured")
        
        # Text input - IMPROVED
        st.subheader("⌨️ Text Input")
        
        with st.form("text_input_form", clear_on_submit=True):
            text_input = st.text_area(
                "Type your message:",
                height=100,
                placeholder="Ask me anything...",
                key="text_input_field"
            )
            
            col_submit1, col_submit2 = st.columns([1, 4])
            with col_submit1:
                submitted = st.form_submit_button("📤 Send", use_container_width=True)
            with col_submit2:
                st.write("")  # Spacer
            
            if submitted and text_input.strip():
                with st.spinner("🤔 Generating response..."):
                    response = st.session_state.assistant.get_response(text_input.strip())
                    
                    voice_response = None
                    if tts_status:  # If any TTS option is available
                        with st.spinner("🔊 Generating voice response..."):
                            voice_response = st.session_state.assistant.generate_voice_response(response)
                    
                    st.session_state.conversation.append(("user", text_input.strip(), None))
                    st.session_state.conversation.append(("assistant", response, voice_response))
                    
                    st.rerun()
    
    with col2:
        st.header("📊 Status")
        
        # Assistant status
        if st.session_state.assistant:
            if st.session_state.assistant.initialization_error:
                st.error("❌ Assistant initialization failed")
                st.error(st.session_state.assistant.initialization_error)
            else:
                st.success("✅ Assistant ready")
                st.info(f"🤖 Model: {st.session_state.assistant.config.model_name}")
                
                if elevenlabs_api_key and AUDIO_AVAILABLE:
                    st.success("🔊 Voice fully enabled (ElevenLabs)")
                elif tts_status and AUDIO_AVAILABLE:
                    st.success(f"🔊 Voice enabled ({', '.join(tts_status[1:]) if len(tts_status) > 1 else tts_status[0]})")
                elif elevenlabs_api_key:
                    st.warning("🔇 ElevenLabs ready, but audio libs missing")
                elif tts_status:
                    st.info(f"🔊 Fallback TTS available ({', '.join(tts_status)})")
                else:
                    st.info("🔇 No voice features available")
                
                if st.session_state.documents_loaded:
                    st.success("📚 Knowledge base loaded")
                else:
                    st.info("📚 No knowledge base")
        else:
            st.warning("⚠️ Assistant not initialized")
        
        # Audio status
        if AUDIO_AVAILABLE:
            st.success("🎵 Audio libraries available")
        else:
            st.error("🚫 Audio libraries missing")
        
        # Controls
        st.header("🎛️ Controls")
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.last_audio_hash = None
            st.session_state.processing_audio = False
            if st.session_state.assistant:
                st.session_state.assistant.memory.clear()
            st.rerun()
        
        if st.button("🔄 Reset Assistant", use_container_width=True):
            st.session_state.assistant = None
            st.session_state.conversation = []
            st.session_state.documents_loaded = False
            st.session_state.last_audio_hash = None
            st.session_state.processing_audio = False
            st.rerun()
        
        
        # Voice usage
        voice_messages = sum(1 for _, _, audio in st.session_state.conversation if audio is not None)
        st.metric("Voice Responses", voice_messages)

if __name__ == "__main__":
    main()
