# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a German language tutor AI voice chat application built with Gradio. It integrates multiple AI services:
- **DeepSeek API**: Text generation for conversational AI responses
- **Google Gemini 2.5 Flash TTS**: Text-to-speech conversion
- **Whisper (large-v3)**: Speech-to-text transcription

The application allows users to practice German conversation through text or voice input, with audio responses.

## Running the Application

```bash
python main.py
```

The application will launch automatically in the browser with Gradio's web interface.

## Environment Setup

Required API keys must be set in `.env` file:
- `DEEPSEEK_API_KEY`: For DeepSeek chat completions
- `GOOGLE_API_KEY`: For Google Gemini TTS

## Architecture

### Core Components

**main.py** (single-file application):
- `chat()` (line 54): Main conversation handler that sends user messages to DeepSeek and generates audio response
- `toAudio()` (line 40): Converts text responses to audio using Google Gemini TTS, saves to `out.wav`
- `audio_to_text()` (line 72): Transcribes audio input using Whisper model
- `put_msg_in_chatbot()` (line 69): Updates chatbot history with user messages

### Key Dependencies

- **gradio**: Web UI framework
- **openai**: Client library for DeepSeek API (OpenAI-compatible)
- **google-genai**: Google Gemini API client
- **openai-whisper**: Speech recognition model
- **soundfile**: Audio file I/O
- **python-dotenv**: Environment variable management

### Application Flow

1. User submits text message or records audio
2. If audio: Whisper transcribes to text → populates text input
3. Text message added to chat history
4. DeepSeek generates German tutor response
5. Response converted to audio via Gemini TTS
6. Audio plays automatically in browser

### Models Used

- **DeepSeek Chat**: `deepseek-chat` model for conversational responses
- **Google Gemini**: `gemini-2.5-flash-preview-tts` for TTS with audio modality
- **Whisper**: `large-v3` model loaded locally for transcription

### Important Notes

- System prompt is hardcoded in `chat()` at line 56 (note: says "French" but should be "German")
- Audio output always saved to `out.wav` in project root
- Whisper model loaded at startup (line 31), which may take time and memory
- Gradio UI configured with autoplay for audio responses
