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


# -------------------- VECTOR STORE --------------------
def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    test_vector = embeddings.embed_query("test")
    st.info(f"✅ Embedding loaded | Dimension: {len(test_vector)}")

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    st.success(f"✅ Vectors created: {vectorstore.index.ntotal}")
    return vectorstore


# -------------------- CONVERSATION CHAIN --------------------
def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# -------------------- HANDLE USER INPUT --------------------
def handle_userinput(user_question):
    response = st.session_state.conversation(
        {"question": user_question}
    )

    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )


# -------------------- MAIN APP --------------------
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("📚 Chat with Multiple PDFs")
    st.write("Upload PDFs from the sidebar and click **Process**")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

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
                    st.session_state.conversation = create_conversation_chain(vectorstore)

                    st.success("🎉 PDFs processed successfully!")


if __name__ == "__main__":
    main()
