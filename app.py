import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter  # Fixed import
# Make sure you have 'langchain' installed and updated
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS


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


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat With Multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
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
            if pdf_docs:  # Fixed extra colon
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in the PDF.")
                    return

                text_chunks = get_text_chunk(raw_text)
                st.write(text_chunks)
            else:
                st.warning("Please upload at least one PDF.")


if __name__ == '__main__':
    main()
