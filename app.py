import logging
import os
import shutil
import uuid
from pathlib import Path

import gradio as gr

import config
import engine
import ingestion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_new_session():
    """Generate a new unique session ID."""
    return str(uuid.uuid4())


def handle_file_upload(files, session_id):
    """Save uploaded files to the session-specific folder."""
    if not files:
        return "No files uploaded."

    # Additional safety: reject files with non-PDF extensions (though Gradio restricts)
    for file in files:
        ext = Path(file.name).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            return f"Unsupported file type: {ext}. Please upload only PDF files."

    upload_path = config.get_session_upload_path(session_id)
    upload_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for file in files:
        dest = upload_path / Path(file.name).name
        shutil.copy(file.name, str(dest))
        saved_count += 1

    return f"Successfully uploaded {saved_count} document(s). Index will be built on first query."


def process_message(message, history, session_id):
    """
    Process user message and get RAG response.
    History format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if not message:
        return "", history

    # Initialize history if None
    if history is None:
        history = []

    # 1. Ensure Index Exists
    store_path = config.get_session_vectorstore_path(session_id)
    if not store_path.exists():
        try:
            ingestion.create_vector_store(session_id)
            bot_response = "Index built successfully! You can now ask questions."
        except Exception as e:
            return "", history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Error building index: {str(e)}"},
            ]

    # 2. Get RAG Response (now returns answer, sources, elapsed_time)
    try:
        answer, sources, elapsed = engine.get_rag_response(session_id, message, history)

        # 3. Format Response with Sources and Time
        if sources:
            source_lines = "\n".join([f"- {s}" for s in sources])
            footer = f"\n\n---\n**Retrieved from:**\n{source_lines}\n\n**Response time:** {elapsed:.2f} s"
            bot_response = f"{answer}{footer}"
        else:
            bot_response = answer

    except Exception as e:
        bot_response = f"Error: {str(e)}"

    # 4. Update History (dict format)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": bot_response},
    ]

    return "", history


def clear_chat():
    """Reset chat history UI."""
    return []


def new_chat_session():
    """Create new session and clear chat."""
    return str(uuid.uuid4()), []


# --- UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Legal Contract RAG Assistant")
    gr.Markdown(
        "Upload employment contracts and ask questions. Each chat session is isolated."
    )

    # State
    session_id = gr.State(value=create_new_session())

    with gr.Row():
        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Documents")
            file_upload = gr.File(
                label="Upload PDFs", file_count="multiple", file_types=[".pdf"]
            )
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

            gr.Markdown("### ⚙️ Session")
            btn_new_chat = gr.Button("🔄 New Chat Session", variant="secondary")
            session_info = gr.Textbox(label="Session ID", interactive=False)

            gr.Markdown("### ℹ️ Info")
            gr.Markdown(
                f"**Model:** {config.OLLAMA_MODEL}\n**Embeddings:** {config.EMBEDDING_MODEL} (CPU)"
            )

        # Chat Area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Contract Q&A",
                height=500,
                value=[],  # Explicitly initialize as empty list
            )

            msg_input = gr.Textbox(
                placeholder="Ask about termination, benefits, etc.",
                label="Your Question",
            )

            with gr.Row():
                send_btn = gr.Button("Send")
                clear_btn = gr.Button("🗑️ Clear Chat")

            # Examples
            gr.Examples(
                examples=[
                    "What is the notice period?",
                    "Explain the confidentiality clause.",
                    "What are the termination conditions?",
                ],
                inputs=msg_input,
            )

    # --- Event Handlers ---
    file_upload.upload(
        fn=handle_file_upload, inputs=[file_upload, session_id], outputs=[upload_status]
    )

    # Chat flow
    msg_input.submit(
        fn=process_message,
        inputs=[msg_input, chatbot, session_id],
        outputs=[msg_input, chatbot],
    )

    send_btn.click(
        fn=process_message,
        inputs=[msg_input, chatbot, session_id],
        outputs=[msg_input, chatbot],
    )

    # New Chat & Clear
    btn_new_chat.click(fn=new_chat_session, inputs=[], outputs=[session_id, chatbot])

    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot])

if __name__ == "__main__":
    logger.info("Starting Legal RAG Application...")
    logger.info(f"Using Model: {config.OLLAMA_MODEL}")
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        logger.error(f"Failed to launch app: {e}")
