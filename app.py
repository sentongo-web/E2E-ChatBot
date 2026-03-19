import streamlit as st
from pathlib import Path

from src.llm_ops import LLMManager


def main():
    st.set_page_config(page_title="Uganda Health Assistant", page_icon="❤️", layout="centered")
    st.title("🇺🇬 Uganda Health Assistant")
    st.write(
        "Ask health and wellness questions in simple language. This assistant is built from open-source LLMs and local knowledge for Uganda context."
    )

    st.sidebar.header("LLM Ops Controls")
    model_option = st.sidebar.selectbox("Choose model", ["google/flan-t5-small", "google/flan-t5-base"])
    max_len = st.sidebar.slider("Answer max tokens", min_value=80, max_value=512, value=220, step=20)
    temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.expander("Why this assistant?", expanded=False):
        st.write(
            "This is a functional demonstration of a health assistant using open-source LLMs, local knowledge retrieval, and prompt design best practices. "
            "We keep it safe and informative, and we always recommend clinical follow-up."
        )

    question = st.text_area("Your health question", placeholder="e.g. How can I manage a fever in my child?", height=100)
    if st.button("Get Advice"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            helper_path = Path("data/uganda_health_knowledge.json")
            if not helper_path.exists():
                st.error("Knowledge data missing. Run setup by creating data file first.")
                return

            llm = LLMManager(model_name=model_option)
            llm.assistant.config.max_length = max_len
            llm.assistant.config.temperature = temp
            response = llm.chat(question)
            st.markdown("### ✅ Health Assistant Response")
            st.write(response.answer)

            st.session_state.history.append(f"Q: {question}")
            st.session_state.history.append(f"A: {response.answer}")

    if st.session_state.history:
        st.markdown("### conversation history")
        for item in st.session_state.history[-8:]:
            if item.startswith("Q:"):
                st.markdown(f"**{item}**")
            else:
                st.write(item)


if __name__ == "__main__":
    main()
