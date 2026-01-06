import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import io
import wave
import numpy as np
import soundfile as sf
import tempfile
from google import genai
from google.genai import types
import whisper
import sqlite3
import re


load_dotenv(override=True)

google_api_key = os.getenv("GOOGLE_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

print(f"deepseek key:{deepseek_api_key}")
print(f"google key:{google_api_key}")

ds_model=OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com",
)

google_model = genai.Client(api_key=google_api_key)
model_id="gemini-2.5-flash-preview-tts"
whisper_model = whisper.load_model("large-v3")

# Initialize SQLite database for dictionary
def init_dictionary_db():
    conn = sqlite3.connect('german_dictionary.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            german_word TEXT UNIQUE NOT NULL,
            english_translation TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_dictionary_db()

# Extract unique words from last AI response only
def extract_words_from_conversation(history):
    """Extract unique German words from the last AI response"""
    words = set()
    # Common English words to exclude
    english_stopwords = {
        'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
        'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'what', 'when', 'where', 'who', 'why', 'how', 'can', 'could',
        'would', 'should', 'will', 'shall', 'may', 'might', 'must',
        'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been',
        'am', 'no', 'yes', 'not', 'so', 'if', 'then', 'else'
    }

    # Only process the last assistant message
    for msg in reversed(history):
        if msg["role"] == "assistant":
            text = msg["content"]
            # Extract words (German words may contain umlauts)
            word_list = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]+\b', text)
            for word in word_list:
                word_lower = word.lower()
                # Filter: length > 1, not English stopword, contains German chars or capitalized
                if (len(word_lower) > 1 and
                    word_lower not in english_stopwords):
                    words.add(word_lower)
            break  # Only process the most recent assistant message
    return sorted(list(words))

# Database operations
def get_translation(german_word):
    """Get English translation for a German word from dictionary"""
    conn = sqlite3.connect('german_dictionary.db')
    cursor = conn.cursor()
    cursor.execute('SELECT english_translation FROM dictionary WHERE german_word = ?', (german_word.lower(),))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def add_translation(german_word, english_translation):
    """Add or update translation in dictionary"""
    conn = sqlite3.connect('german_dictionary.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO dictionary (german_word, english_translation)
        VALUES (?, ?)
        ON CONFLICT(german_word) DO UPDATE SET english_translation = ?
    ''', (german_word.lower(), english_translation, english_translation))
    conn.commit()
    conn.close()

def get_vocabulary_display(history):
    """Get vocabulary list with translations for display"""
    words = extract_words_from_conversation(history)
    vocab_list = []
    for word in words:
        translation = get_translation(word)
        vocab_list.append({
            "german": word,
            "english": translation if translation else "Not in dictionary"
        })
    return vocab_list

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

def toAudio(text):
    response = google_model.models.generate_content(
        model=model_id,
        contents=f"Say '{text}'",
        config={"response_modalities":['Audio']},
    )

    blob = response.candidates[0].content.parts[0].inline_data.data
    file_name = 'out.wav'
    wave_file(file_name, blob)
    #os.path.dirname(__file__))
    return os.path.dirname(__file__) + "/out.wav"


def chat(history):
    history = [{"role":h["role"],"content":h["content"]} for h in history]
    context_message ="You are German tutor who practice german conversation with me, but also understand English, in case the learner ask question in English, you translate it first then reply in German"
    messages = [{"role":"system", "content":context_message}] + history
    response =ds_model.chat.completions.create(
        messages=messages,
        model="deepseek-chat",
        stream=False
    )
    res= response.choices[0].message.content
    voice = toAudio(res)
    # play_audio_blob(voice)
    history += [{"role":"assistant","content":res}]

    # Get vocabulary list for display
    vocab_list = get_vocabulary_display(history)
    vocab_df = [[v["german"], v["english"]] for v in vocab_list]

    return history, voice, vocab_df

def put_msg_in_chatbot(message,history):
    return "", history + [{"role":"user","content":message}]

def audio_to_text(audio):
    if audio is None:
        return "No Audio Provided"
    sr, audio_data = audio #sample rate (SR) and audio data (audio_data)

    # For stereo audio, convert to mono (Whisper expects mono)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Convert numpy array to proper format and save
        import soundfile as sf
        sf.write(f.name, audio_data, sr)
        # Transcribe
        result = whisper_model.transcribe(f.name)
        print(result["text"])
    return result["text"]

def save_translation(german_word, english_translation, history):
    """Save translation and refresh vocabulary display"""
    if german_word and english_translation:
        add_translation(german_word, english_translation)
        vocab_list = get_vocabulary_display(history)
        vocab_df = [[v["german"], v["english"]] for v in vocab_list]
        return vocab_df, "", ""
    return None, german_word, english_translation


with (gr.Blocks() as ui):
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, type="messages")
            with gr.Row():
                audio_output = gr.Audio(autoplay=True, label="Voice Playback")
            with gr.Row():
                message = gr.Textbox(label="Chat with AI with Text")
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Speak to the AI"
                )

        with gr.Column(scale=1):
            gr.Markdown("## Vocabulary from Last Response")
            vocab_table = gr.Dataframe(
                headers=["German Word", "English Translation"],
                datatype=["str", "str"],
                label="Dictionary",
                interactive=False,
                wrap=True
            )
            gr.Markdown("### Add New Translation")
            german_input = gr.Textbox(label="German Word", placeholder="Enter German word")
            english_input = gr.Textbox(label="English Translation", placeholder="Enter English translation")
            save_btn = gr.Button("Save Translation")

    audio_input.stop_recording(audio_to_text,inputs=[audio_input], outputs=[message]).then(fn = lambda: None, inputs=None,).then(None,None,None)

    message.submit(put_msg_in_chatbot,inputs=[message,chatbot], outputs=[message,chatbot]).then(
        chat, inputs=chatbot,outputs=[chatbot, audio_output, vocab_table]
    )

    save_btn.click(
        save_translation,
        inputs=[german_input, english_input, chatbot],
        outputs=[vocab_table, german_input, english_input]
    )

ui.launch(inbrowser=True)


