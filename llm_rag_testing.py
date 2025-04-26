
import os
import base64
import uuid
from dotenv import load_dotenv, find_dotenv
import pytesseract

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from unstructured.partition.pdf import partition_pdf

# Step 1: Load environment variables
_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 2: Initialize LLM models
chain_gpt_35 = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
chain_gpt_4o = ChatOpenAI(model="gpt-4o", max_tokens=1024)

# Step 3: Set Tesseract OCR path (important for image extraction)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Step 4: Setup input and output directories
input_path = os.getcwd()
output_path = os.path.join(os.getcwd(), "figures")

os.makedirs(output_path, exist_ok=True)

# Step 5: Parse PDF to extract text, tables, images
raw_pdf_elements = partition_pdf(
    filename=os.path.join(input_path, "your_pdf_file.pdf"),  # Change filename here
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=output_path,
)

# Step 6: Separate elements into text, tables, images
text_elements = []
table_elements = []
image_elements = []

for element in raw_pdf_elements:
    if 'CompositeElement' in str(type(element)):
        text_elements.append(element.text)
    elif 'Table' in str(type(element)):
        table_elements.append(element.text)

# Encode extracted images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

for image_file in os.listdir(output_path):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(output_path, image_file)
        image_elements.append(encode_image(image_path))

# Step 7: Define summarization functions
def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

def summarize_table(table):
    prompt = f"Summarize the following table:\n\n{table}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

def summarize_image(encoded_image):
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contents of this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ])
    ]
    response = chain_gpt_4o.invoke(prompt)
    return response.content

# Step 8: Summarize all extracted elements
text_summaries = [summarize_text(te) for te in text_elements]
table_summaries = [summarize_table(te) for te in table_elements]
image_summaries = [summarize_image(ie) for ie in image_elements]

# Step 9: Initialize Chroma Vector Database and Docstore
vectorstore = Chroma(collection_name="multimodal_rag_project", embedding_function=OpenAIEmbeddings())
docstore = InMemoryStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=docstore, id_key=id_key)

# Step 10: Add summarized documents to the retriever
def add_documents_to_retriever(summaries, originals, label):
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [
        Document(page_content=f"[{label}] {summaries[i]}", metadata={id_key: doc_ids[i]})
        for i in range(len(summaries))
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, originals)))

add_documents_to_retriever(text_summaries, text_elements, "text")
add_documents_to_retriever(table_summaries, table_elements, "table")
add_documents_to_retriever(image_summaries, image_summaries, "image")

# Step 11: Setup final QA Chain
prompt_template = """Answer the question based only on the following context, which can include text, images, and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
qa_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | qa_model
    | StrOutputParser()
)

# Step 12: Example Queries
print("\nExample Query 1: What products are displayed in the images?")
print(qa_chain.invoke("What products are displayed in the images?"))

print("\nExample Query 2: What is the financial summary of the company?")
print(qa_chain.invoke("What is the financial summary of the company?"))

print("\nExample Query 3: What is the total revenue mentioned?")
print(qa_chain.invoke("What is the total revenue mentioned?"))
