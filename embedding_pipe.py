from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()


def build_vector_store(books_dir: str, persistent_directory: str):
    books_path = Path(books_dir)
    db_path = Path(persistent_directory)

    if not db_path.exists():

        print("Persistent directory does not exist. Initializing vector store...")

        if not books_path.exists():
            raise FileNotFoundError(

                f"The directory {books_path} does not exist. Please check the path."
            )
        
        db_path.mkdir(parents=True, exist_ok=True)

        docs = []

        for path in books_path.glob("*.txt"):

            text = path.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"story_title": path.stem}))
        
        

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        print("Number of documents after splitting: ",len(split_docs))
        print("Sample split document:\n",split_docs[:2])


        embeddings = MistralAIEmbeddings()
        vectordb = Chroma.from_documents(
            split_docs,
            embedding=embeddings,
            persist_directory=str(db_path),
        )
        vectordb.persist()
        return vectordb
    
    else:
        print("Vector store already exists. Loading existing store...")
        embeddings = MistralAIEmbeddings()
        vectordb = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
        )
        return vectordb
