# 🏥 Medical Summarization Bot

## try this directly https://huggingface.co/spaces/Imsachinsingh00/medical-summary-app

This is a simple Gradio-based application for **summarizing medical reports and doctor-patient conversations**. It uses:

- 🎙️ [Whisper](https://github.com/openai/whisper) (base) for automatic speech recognition (ASR)
- 🧠 [BART-Large-CNN](https://huggingface.co/facebook/bart-large-cnn) for text summarization

---

## 🚀 Features

- Upload or record **audio** of medical conversations → get transcription + summary
- Paste **text** from reports or dialogues → get instant summary
- Lightweight, open-source models (runs on CPU)

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/imsachinsingh00/medical-summarization-bot.git
cd medical-summarization-bot
