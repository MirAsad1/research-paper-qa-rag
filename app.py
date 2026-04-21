import os
import streamlit as st
from rag.chain import build_chain

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📄",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("📄 Research Paper Q&A")
    st.markdown("---")

    st.markdown("### 🤖 Model Info")
    st.markdown("- **LLM:** Llama 3 (8B)")
    st.markdown("- **Provider:** Groq")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector Store:** ChromaDB")

    st.markdown("---")

    st.markdown("### 📁 Papers Loaded")
    papers_dir = "data/research_papers"
    papers = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    if papers:
        for paper in papers:
            st.markdown(f"- 📄 {paper}")
    else:
        st.warning("No papers found in data/research_papers/")

    st.markdown("---")

    st.markdown("### 💡 How to Use")
    st.markdown("""
1. Papers are already ingested
2. Type your question below
3. AI will search relevant chunks
4. Sources shown below each answer
""")

    st.markdown("---")
    st.caption("Built with LangChain, Groq & Streamlit")

# --- Main Area ---
st.title("Ask About Your Research Papers")
st.caption("Powered by RAG — Retrieval Augmented Generation")

@st.cache_resource
def load_chain():
    return build_chain()

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the research papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            result = chain({"query": prompt})
            answer = result["result"]
            sources = result["source_documents"]

        st.markdown(answer)

        with st.expander("📚 Sources"):
            seen = set()
            for doc in sources:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                key = f"{source}-{page}"
                if key not in seen:
                    seen.add(key)
                    st.markdown(f"- **{source}** — Page {page}")

    st.session_state.messages.append({"role": "assistant", "content": answer})