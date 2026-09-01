
import os

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from htmlTemplates import css, bot_template, user_template


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_files):
    """Extract text from multiple PDF files."""

    text = ""

    for pdf in pdf_files:
        reader = PdfReader(pdf)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text_into_chunks(text):
    """Split extracted PDF text into smaller chunks."""

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return splitter.split_text(text)


# ============================================================
# VECTOR STORE
# ============================================================

def create_vectorstore(text_chunks):
    """Create FAISS vector database using HuggingFace embeddings."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Test embedding
    test_vector = embeddings.embed_query("test")

    st.info(
        f"✅ Embedding loaded | Dimension: {len(test_vector)}"
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    st.success(
        f"✅ Vectors created: {vectorstore.index.ntotal}"
    )

    return vectorstore


# ============================================================
# CONVERSATION CHAIN
# ============================================================

def create_conversation_chain(vectorstore):
    """Create conversational RAG chain using OpenAI."""

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "❌ OPENAI_API_KEY was not found. "
            "Please add it to your .env file."
        )
        st.stop()

    # OpenAI Chat Model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key
    )

    # Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Conversational Retrieval Chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    return conversation_chain


# ============================================================
# HANDLE USER INPUT
# ============================================================

def handle_userinput(user_question):
    """Send user question to the conversational RAG chain."""

    try:
        response = st.session_state.conversation(
            {"question": user_question}
        )

        st.session_state.chat_history = response["chat_history"]

        # Display conversation
        for i, message in enumerate(
            st.session_state.chat_history
        ):

            if i % 2 == 0:
                st.write(
                    user_template.replace(
                        "{{MSG}}",
                        message.content
                    ),
                    unsafe_allow_html=True
                )

            else:
                st.write(
                    bot_template.replace(
                        "{{MSG}}",
                        message.content
                    ),
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"❌ Error while answering question: {e}")


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # Page configuration
    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon="📚"
    )

    # Load custom CSS
    st.write(
        css,
        unsafe_allow_html=True
    )

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ========================================================
    # MAIN UI
    # ========================================================

    st.header("📚 Chat with Multiple PDFs")

    st.write(
        "Upload PDFs from the sidebar and click **Process**."
    )

    # User question
    user_question = st.text_input(
        "Ask a question about your documents:"
    )

    if (
        user_question
        and st.session_state.conversation
    ):
        handle_userinput(user_question)

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.subheader("📄 Your Documents")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process"):

            if not pdf_docs:

                st.warning(
                    "⚠️ Please upload at least one PDF."
                )

            else:

                with st.spinner(
                    "Processing PDFs..."
                ):

                    try:

                        # Step 1: Extract text
                        raw_text = extract_pdf_text(
                            pdf_docs
                        )

                        if not raw_text.strip():

                            st.error(
                                "❌ Could not extract text "
                                "from the uploaded PDFs."
                            )

                            st.stop()

                        # Step 2: Split text
                        text_chunks = (
                            split_text_into_chunks(
                                raw_text
                            )
                        )

                        st.write(
                            f"📄 Text chunks created: "
                            f"{len(text_chunks)}"
                        )

                        # Step 3: Create vector store
                        vectorstore = create_vectorstore(
                            text_chunks
                        )

                        # Step 4: Create conversation chain
                        st.session_state.conversation = (
                            create_conversation_chain(
                                vectorstore
                            )
                        )

                        # Clear old chat history
                        st.session_state.chat_history = []

                        st.success(
                            "🎉 PDFs processed successfully!"
                        )

                    except Exception as e:

                        st.error(
                            f"❌ Processing failed: {e}"
                        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()