import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # ✅ Load local HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ PROOF 1: Generate a test embedding
    test_vector = embeddings.embed_query("vector embedding test")
    st.info(f"✅ Embedding model loaded | Vector dimension: {len(test_vector)}")
    # Expected output: 384

    # ✅ Create FAISS vector store
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    # ✅ PROOF 2: Show number of vectors created
    st.success(f"✅ Vector embeddings created: {vectorstore.index.ntotal}")

    return vectorstore


def main():
    st.set_page_config(page_title="Chat With Multiple PDFs", page_icon="📚")
    st.header("Chat with multiple PDFs 📚")
    st.write("Upload PDFs from the sidebar and click Process")

    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs and creating embeddings..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)

                    # Split text into chunks
                    text_chunks = get_text_chunk(raw_text)

                    # ✅ PROOF 3: Show chunk count
                    st.write(f"📄 Total text chunks created: {len(text_chunks)}")

                    # Create vector store (embeddings happen here)
                    vectorstore = get_vectorstore(text_chunks)

                    # Store in session
                    st.session_state.vectorstore = vectorstore

                    st.success("🎉 PDFs processed & embeddings stored successfully!")
            else:
                st.warning("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
