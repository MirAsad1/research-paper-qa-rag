from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_retriever():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 9},
    )
    return retriever

