# VoiceAgentTask-TejaswiMahadev

# 🎤 AI Voice Assistant
A powerful voice-enabled AI assistant built using Streamlit, Google Gemini, and ElevenLabs. It supports both text and voice interactions, knowledge retrieval from websites and PDFs, and multiple Text-to-Speech (TTS) options.

## 🚀 Features
-🎙️ Voice Input using your microphone (requires audio libraries)
- 💬 Conversational AI powered by Gemini Flash or Pro

## 🔊 Text-to-Speech using:

- ElevenLabs (API key required)
- Local TTS (pyttsx3)
- Google TTS (gTTS)

## 📚 Knowledge Augmentation with:

- Website scraping
- PDF upload and chunking

## 🧠 Simple memory for maintaining conversation history

## 📊 Real-time status and debugging information

## 📁 Modular code with classes for processing audio, documents, speech, and memory

🧰 Tech Stack

- Web Framework	Streamlit
- LLM Backend	Google Gemini (via LangChain)
- TTS Engine	ElevenLabs, pyttsx3, gTTS
- STT Engine	SpeechRecognition (Google API)
- Audio Input	audio-recorder-streamlit
- Knowledge RAG	FAISS + LangChain document loaders
- File Parsing	PyPDFLoader
