import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Page Configuration setup
st.set_page_config(page_title="RLHF AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Independent AI Assistant Website")
st.caption("Hosted in the cloud — Accessible anywhere via browser.")

# 2. Cache the heavy LLM Pipeline loading function on the cloud server
@st.cache_resource
def load_llm_pipeline():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

ai_generator = load_llm_pipeline()

# 3. Handle Persistent Interface Memory Context States
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your cloud-hosted AI Assistant. Ask me anything!"}]

# Render active window messages 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4. Handle Real-Time Chat Inputs
if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Trigger Text generation parameters
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_format = f"<|im_start|>system\nYou are a helpful and intelligent AI assistant.<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            outputs = ai_generator(
                chat_format, 
                max_new_tokens=200,     
                do_sample=False,        
                repetition_penalty=1.2  
            )
            
            generated_text = outputs[0]["generated_text"]
            clean_response = generated_text.split("<|im_start|>assistant\n")[-1].strip()
            clean_response = clean_response.replace("<|im_end|>", "").strip()
            
            st.write(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})
cache: 'pip'
