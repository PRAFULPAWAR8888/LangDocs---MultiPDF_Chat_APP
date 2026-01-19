import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

from htmlTemplates import css, bot_template, user_template


# -------------------- PDF TEXT EXTRACTION --------------------
def extract_pdf_text(pdf_files):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -------------------- TEXT CHUNKING --------------------
def split_text_into_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


# -------------------- VECTOR STORE CREATION --------------------
def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Test embedding
    test_vector = embeddings.embed_query("test embedding")
    st.info(f"✅ Embedding model loaded | Vector size: {len(test_vector)}")

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    st.success(f"✅ Total vectors created: {vectorstore.index.ntotal}")
    return vectorstore


# -------------------- CONVERSATION CHAIN --------------------
def create_conversation_chain(vectorstore):

    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# -------------------- USER INPUT HANDLING --------------------
def handle_userinput(user_question):
    response = st.session_state.conversation.run({'question': user_question})
    st.write(response)


# -------------------- MAIN APP --------------------
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("📚 Chat with Multiple PDFs")
    st.write("Upload PDF files from the sidebar and click **Process**")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello, Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello, Human"), unsafe_allow_html=True)

    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        st.subheader("Your Documents")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = extract_pdf_text(pdf_docs)
                    text_chunks = split_text_into_chunks(raw_text)

                    st.write(f"📄 Text chunks created: {len(text_chunks)}")

                    vectorstore = create_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation = create_conversation_chain(vectorstore)

                    st.success("🎉 PDFs processed successfully!")

    # Ensure conversation chain exists
    if (
        "vectorstore" in st.session_state
        and st.session_state.vectorstore is not None
        and st.session_state.conversation is None
    ):
        st.session_state.conversation = create_conversation_chain(
            st.session_state.vectorstore
        )


if __name__ == "__main__":
    main()
