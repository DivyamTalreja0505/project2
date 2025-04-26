# llm_rag_testing.py

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid

# Load OpenAI API key
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Initialize models
chain_gpt_35 = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
chain_gpt_4o = ChatOpenAI(model="gpt-4o", max_tokens=1024)

# Example texts
texts = [
    "Artificial Intelligence is transforming industries through automation and data insights.",
    "Retrieval-Augmented Generation combines document retrieval and language modeling for improved question answering."
]

# Summarization function
def summarize_text(model, text):
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    response = model.invoke([{"role": "user", "content": prompt}])
    return response.content

# Summarize texts using both models
summaries_gpt_35 = []
summaries_gpt_4o = []

for text in texts:
    summaries_gpt_35.append(summarize_text(chain_gpt_35, text))
    summaries_gpt_4o.append(summarize_text(chain_gpt_4o, text))

# Initialize database and retriever
vectorstore = Chroma(collection_name="llm_rag_project", embedding_function=OpenAIEmbeddings())
docstore = InMemoryStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=docstore, id_key=id_key)

# Add documents to retriever
def add_documents_to_retriever(summaries, originals, label):
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [
        Document(page_content=f"[{label}] {summaries[i]}", metadata={id_key: doc_ids[i]})
        for i in range(len(summaries))
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, originals)))

add_documents_to_retriever(summaries_gpt_35, texts, "gpt-3.5-turbo")
add_documents_to_retriever(summaries_gpt_4o, texts, "gpt-4o")

# Test retrieval
print("\nRetrieving documents related to RAG...\n")
retrieved_docs = retriever.invoke("Explain RAG in simple terms.")

for idx, doc in enumerate(retrieved_docs):
    print(f"Document {idx+1}: {doc}\n")
