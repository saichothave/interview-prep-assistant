# 🎯 Interview Prep Assistant (Backend)

This is the backend server for the **Resume Bullet & Interview Prep Assistant**. It generates high-impact resume bullet points and relevant technical interview questions from user-provided experience stories using the Mistral language model via [Ollama](https://ollama.com/).

## 🚀 Features

- Converts experience stories into resume bullet points using the what–how–effect structure.
- Suggests 3–5 interview questions based on the generated resume bullet.
- Accepts optional user instructions to customize tone, structure, or content.
- Provides a contextual chatbot-style Q&A experience based on the generated outputs.
- FastAPI-powered API with CORS enabled for frontend integration.
- Compatible with Ollama local model hosting (e.g., Mistral).

---

## 🛠️ Tech Stack

- Python 3.10+
- FastAPI
- Ollama (running Mistral or other LLMs locally)
- Pydantic
- CORS Middleware

---
