import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set Hugging Face Token (Replace with your own token)
HUGGINGFACE_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN

# Load and process the PDF
def load_documents(pdf_path):
    """Loads and splits documents into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100  # Overlapping context for better retrieval
    )

    docs = text_splitter.split_documents(documents)
    return docs

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Initialize the LLM
llm = llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    model_kwargs={"temperature": 0.7, "max_length": 512},
    huggingfacehub_api_token="hf_waeuMsYDlfeKvicDXbGDcIVcQvuIckIUCYC:\Users\sribh\OneDrive\Desktop\VS CODE FILES\rag_floder\.venv\ncert maths.pdf"  # Replace with your token
)

# RAG Pipeline with Memory
def build_rag_pipeline(pdf_path):
    """Creates the vector store and retrieval-based chatbot pipeline with memory."""
    docs = load_documents(pdf_path)

    # Create FAISS Vector Store
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # Conversation Memory (to remember past interactions)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory=memory
    )

    return qa_chain

# Function to interact with the chatbot
def chat_with_rag(qa_chain):
    """Starts an interactive chat session with the RAG chatbot."""
    print("\nChatbot is ready! Ask questions about the document (type 'exit' to quit).")
    chat_history = []

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("Exiting chatbot...")
            break

        response = qa_chain.invoke({"question": query, "chat_history": chat_history})
        chat_history.append((query, response["answer"]))

        print("\nChatbot:", response["answer"])

# Main function
if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF document: ")
    if not os.path.exists(pdf_path):
        print("Invalid file path. Please provide a valid PDF document.")
    else:
        print("\nBuilding RAG Chatbot...")
        qa_chain = build_rag_pipeline(pdf_path)
        chat_with_rag(qa_chain)