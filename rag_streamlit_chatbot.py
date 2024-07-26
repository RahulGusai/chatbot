import os
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

# Initialize API and models
api_key = "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"
llm = ChatAnthropic(model="claude-3-sonnet-20240229",
                    anthropic_api_key=api_key)

# Load data from a URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=HuggingFaceEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Streamlit app
st.title("Chatbot")
if "history" not in st.session_state:
    st.session_state.history = []


def get_response(question):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(question)
    return response


user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        st.session_state.history.append(
            {"role": "user", "content": user_input})
        response = get_response(user_input)
        print(response)
        st.session_state.history.append(
            {"role": "assistant", "content": response})

# Display conversation history
for message in st.session_state.history:
    if message["role"] == "user":
        st.text(f"You: {message['content']}")
    else:
        st.text(f"Assistant: {message['content']}")
