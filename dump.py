

# store = {}
# config = {"configurable": {"session_id": "abc1"}}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# with_message_history = RunnableWithMessageHistory(
#     chain, get_session_history, input_messages_key="messages")

# response = chain.invoke(
#     {"messages": [HumanMessage(content=prompt)],
#      "language": "english"},
# )

# with st.chat_message("assistant"):
#     st.markdown(response.content)
# st.session_state.messages.append(
#     {"role": "user", "content": response.content})
