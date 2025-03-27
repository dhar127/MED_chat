from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Load PDF files from a directory
path = r"C:\Users\dhara\OneDrive\Desktop\MED_chat\data"  # Use absolute path

def load_pdfs(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

documents = load_pdfs(path)
#print("Number of documents loaded:", len(documents))

#create chunks
def chunks(extracted_data):
    text_split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_split.split_documents(extracted_data)
    return text_chunks
text_chunks=chunks(extracted_data=documents)
#print("Length of text chunks:",len(text_chunks))

#vector embeddings
def getembedding():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model=getembedding()

#store embedding in FAISS
db_faiss_path="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(db_faiss_path)