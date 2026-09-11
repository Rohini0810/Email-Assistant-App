# Local AI Email Assistant using LangChain and Ollama
import streamlit as st
from langchain_ollama import ChatOllama


st.set_page_config(
    page_title="Local AI Email Assistant",
    page_icon="✉️",
    layout="centered",
)

st.title("✉️ Local AI Email Responder")

st.write(
    "Turn your rough notes into a professional email, "
    "running 100% locally."
)


tone = st.selectbox(
    "Select Email Tone",
    ["Professional", "Friendly", "Apologetic", "Direct"],
)

user_input = st.text_area(
    "Enter your rough draft or notes here:",
    height=150,
)


llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
)


if st.button("Generate Professional Email"):

    if not user_input.strip():

        st.warning("Please enter some text to get started.")

    else:

        with st.spinner("Drafting your email..."):

            system_prompt = f"""
            You are an expert corporate communicator.

            Your task is to rewrite the user's rough notes
            into a well-structured, grammatically correct email.

            The tone of the email must be: {tone}.

            Do not include any explanations or pleasantries.
            Return only the email itself.
            """

            try:

                response = llm.invoke([
                    ("system", system_prompt),
                    ("human", user_input)
                ])

                st.subheader("Your Polished Email:")
                st.info(response.content)

            except Exception as e:

                st.error(
                    f"An error occurred: {e}. "
                    "Make sure Ollama is running in Docker."
                )