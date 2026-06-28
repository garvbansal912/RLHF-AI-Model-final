import os
import sqlite3
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Initialize local Database configuration tracking logs
DB_PATH = "Garv/chat_telemetry.db"
os.makedirs("Garv", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# 2. Page Configuration setup
st.set_page_config(page_title="RLHF AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Custom RLHF-Aligned Assistant")
st.caption("Running high-intelligence Instruct weights locally on CPU.")

# --- NEW FEATURE: DYNAMIC WORD LIMIT SIDEBAR CONTROL ---
with st.sidebar:
    st.header("⚙️ Response Settings")
    st.write("Control your assistant's maximum sentence structure outputs.")
    
    # Creates a visual slider allowing any value between 50 and 500 words
    max_words = st.slider(
        label="Maximum Word Limit",
        min_value=50,
        max_value=500,
        value=150,  # Starts at a comfortable 150-word middle ground
        step=25,
        help="Higher values allow detailed essays, lower values keep answers short and fast."
    )
# -----------------------------------------------------

# 3. Cache the heavy LLM Pipeline loading function
@st.cache_resource
def load_llm_pipeline():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

ai_generator = load_llm_pipeline()

# 4. Handle Persistent Interface Memory Context States
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your custom AI Assistant. Adjust the slider on the left sidebar to control how long my responses should be!"}]

# Render active window messages 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 5. Handle Real-Time Chat Inputs
if user_input := st.chat_input("Message your assistant..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Trigger Text generation parameters
    with st.chat_message("assistant"):
        with st.spinner("Processing generation checks..."):
            
            # Formulate prompt injecting the dynamic sidebar choice explicitly into system guidance
            chat_format = (
                f"<|im_start|>system\nYou are a helpful and intelligent AI assistant. "
                f"Provide a clear answer to the user's request. Keep your entire output concise and restricted "
                f"to roughly under {max_words} words total.<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            )
            
            # Map the slider choice directly into max_new_tokens
            outputs = ai_generator(
                chat_format, 
                max_new_tokens=max_words,     # Dynamic token allocation matching your slider value!
                do_sample=False,        
                repetition_penalty=1.2  
            )
            
            generated_text = outputs[0]["generated_text"]
            
            # Formulating clean output blocks
            clean_response = generated_text.split("<|im_start|>assistant\n")[-1].strip()
            clean_response = clean_response.replace("<|im_end|>", "").strip()
            
            if not clean_response:
                clean_response = "I am ready to help you with your tasks!"
            
            st.write(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})

    # Log variables directly into your local database tracker
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversation_logs (prompt, response) VALUES (?, ?)", (user_input, clean_response))
    conn.commit()
    conn.close()
