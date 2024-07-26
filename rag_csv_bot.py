import os
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize API and models
api_key = "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"
llm = ChatAnthropic(model="claude-3-sonnet-20240229",
                    anthropic_api_key=api_key)

# Load data from a CSV file
csv_file_path = 'path/to/your/file.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Assuming the relevant text is in a column named 'content'
texts = df['content'].tolist()

# Split text into documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = [text_splitter.split_text(text) for text in texts]

# Flatten the list of lists into a single list of documents
splits = [doc for sublist in splits for doc in sublist]

# Create the vector store
vectorstore = Chroma.from_documents(
    documents=splits, embedding=HuggingFaceEmbeddings())

# Retrieve and generate using the relevant snippets
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is task decomposition?")
print(response)
