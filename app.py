import torch
import gradio as gr
import whisper
from transformers import pipeline

# 🔹 Load Whisper model (base) for transcription
model_whisper = whisper.load_model("base")

# 🔹 Load summarization pipeline using BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 🔹 Transcription + Summarization
def transcribe_and_summarize(audio_file):
    try:
        result = model_whisper.transcribe(audio_file)
        transcription = result.get("text", "")
        if not transcription.strip():
            return "No speech detected", "Summary not generated"
        summary = summarizer(transcription, max_length=100, min_length=30, do_sample=False)
        return transcription, summary[0]['summary_text']
    except Exception as e:
        return "Transcription failed", f"Error: {str(e)}"

# 🔹 Text-only summarization
def summarize_text_input(text_input):
    if not text_input.strip():
        return "", "Input is empty"
    summary = summarizer(text_input, max_length=100, min_length=30, do_sample=False)
    return text_input, summary[0]['summary_text']

# 🔹 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🏥 Medical Summarization App (Whisper + BART)")

    with gr.Tab("🎙️ Record & Summarize"):
        audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")
        mic_transcript = gr.Textbox(label="Transcript")
        mic_summary = gr.Textbox(label="Summary", interactive=False)
        mic_button = gr.Button("Transcribe & Summarize")
        mic_button.click(transcribe_and_summarize, inputs=[audio_input], outputs=[mic_transcript, mic_summary])

    with gr.Tab("📋 Paste & Summarize"):
        text_input = gr.Textbox(lines=8, label="Paste Medical Report or Dialogue")
        text_output = gr.Textbox(label="Summary", interactive=False)
        text_button = gr.Button("Summarize")
        text_button.click(summarize_text_input, inputs=[text_input], outputs=[text_input, text_output])

demo.launch(share=True)
