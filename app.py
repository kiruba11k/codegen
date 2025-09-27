import os
import streamlit as st
from groq import Groq
from e2b_code_interpreter import Sandbox  # optional, if you want to run code

# Load environment
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in environment or secrets.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Groq Analyzer + CodeGen", layout="wide")

st.title("Groq-powered Analyzer & Code Generator")

# Sidebar: choose mode
mode = st.sidebar.radio("Mode", ["Analyze / Text", "Code Generation"])

# Common parameters
with st.sidebar.expander("Model / Parameters"):
    model = st.text_input("Groq model name", value="qwen-2.5-coder-32b")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.number_input("Max completion tokens", min_value=32, max_value=2048, value=512, step=32)
    # You can add more parameters like top_p, stop sequences, etc.

def groq_chat_completion(messages, stream=False):
    """Helper to call Groq chat completions."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=stream,
    )
    return resp

def parse_streaming(resp):
    """Yield text from streaming response chunks."""
    for chunk in resp:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                yield delta.content

if mode == "Analyze / Text":
    st.subheader("Text Analysis / Processing")
    input_text = st.text_area("Enter the text you want analyzed", height=200)
    # Optional: choose analysis type
    analysis_type = st.selectbox("Analysis type", ["Summarize", "Sentiment", "Extract Keywords", "Explain"])

    if st.button("Analyze"):
        # build prompt
        system_prompt = "You are a helpful assistant specialized in text analysis."
        user_prompt = f"Please {analysis_type.lower()} the following text:\n\n{input_text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        with st.spinner("Calling Groq..."):
            resp = groq_chat_completion(messages, stream=False)
        result = resp.choices[0].message.content
        st.markdown("**Result:**")
        st.write(result)

elif mode == "Code Generation":
    st.subheader("Code Generation from Description")
    desc = st.text_area("Enter the problem / specification you want code for", height=200)
    language = st.selectbox("Programming language", ["python", "javascript", "sql", "r", "bash"])
    run_code = st.checkbox("Execute code (in sandbox)", value=False)

    if st.button("Generate Code"):
        system_prompt = "You are an expert coder and generate clean, commented, correct code."
        user_prompt = f"Write {language} code for the following:\n\n{desc}\n\n# Provide working code with comments."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        with st.spinner("Generating code via Groq..."):
            resp = groq_chat_completion(messages, stream=False)
        code = resp.choices[0].message.content
        st.markdown("**Generated Code:**")
        st.code(code, language=language)

        if run_code:
            st.markdown("**Execution Output (sandbox):**")
            try:
                sbx = Sandbox()
                output = sbx.run(code, language=language)
                st.write(output)
            except Exception as e:
                st.error(f"Error running code: {e}")

# Optionally, show session state, debug info, etc.
