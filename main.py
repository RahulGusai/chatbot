from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import streamlit as st

api_key = "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"
llm = ChatAnthropic(model="claude-3-sonnet-20240229",
                    anthropic_api_key=api_key)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(
            {"role": "user", "content": prompt})

    message_container = st.chat_message("assistant")
    message_placeholder = message_container.empty()

    # Stream the response
    response_content = ""
    for response in chain.stream({"messages": [HumanMessage(content=prompt)],
                                  "language": "english"}):
        response_content += response.content
        message_placeholder.markdown(response_content)
    st.session_state.messages.append(
        {"role": "assistant", "content": response_content})
