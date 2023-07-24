import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,ConversationChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text  
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, 
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
def get_conversation(vectorstore):
    llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    DEFAULT_TEMPLATE = """System: You are the assistant of a software company.
        Your role is to conduct initial interviews for job applicants.
        Do not get into details. 
        Go over his/her cv which is provided by following text "{context}".
        Ask questions about the information provided in the cv.
        At the end of each conversation, create a report for the conversation"
        Do not answer the questions that you do not know.   
        {chat_history}
        Human: {question}
        Assistant:"""
    PROMPT = PromptTemplate(input_variables=["context", "question","chat_history"], template=DEFAULT_TEMPLATE)
    condense_prompt = PromptTemplate.from_template(
        ("""Return the following text as it is
         "{question}"
         """)
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt":PROMPT},
        verbose=True
    )

    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st. session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
def main():
    st.set_page_config(page_title="Interview bot", page_icon=":robot_face:")

    st.header("Interview Bot")
    user_question = st.text_input("How can I help you?")
    if user_question:
        handle_user_input(user_question)
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    
    with st.sidebar:
        st.subheader("Your CV")
        pdf_docs=st.file_uploader("Upload your CV here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # Create vector store with embeddings
                vectorstore = get_vectorstore(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation(vectorstore)
    
if __name__ == '__main__':
    main()