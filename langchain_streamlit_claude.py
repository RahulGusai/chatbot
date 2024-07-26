import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
import os

os.environ["ANTHROPIC_API_KEY"] = "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"

model = ChatAnthropic(model="claude-3-sonnet-20240229")
conversation = ConversationChain(llm=model)

st.title("Claude Chatbot")
st.write("Chat with Claude, the large language model from Anthropic!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.write(f"**{message['role']}**: {message['content']}")

user_input = st.text_input("You: ")

if user_input:
    st.session_state.messages.append({"role": "User", "content": user_input})

    response = conversation.run(user_input)

    st.markdown(response)

    st.session_state.messages.append({"role": "Claude", "content": response})
