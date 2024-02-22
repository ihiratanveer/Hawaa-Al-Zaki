import streamlit as st


st.title('🦜🔗 Hawaar-al-Zaki')

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


embeddings = HuggingFaceInstructEmbeddings()
db = FAISS.load_local("db", embeddings)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
#repo_id = "mlabonne/NeuralMonarch-7B"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 1024})

system_prompt = """You are a Muslim Scholar who helps the Muslilms with their queries related to Umrah.
        If you don't know the answer to a ny question from the documents provided to you, then apologize. Give maximum context of your answer.
        Given the Context and chat history answer the user questions."""
B_INST, E_INST = "<s>[INST] ", " [/INST]"
template = (
                B_INST
                + system_prompt
                + """

            Context: {context} / {chat_history}
            User: {question}"""
                + E_INST
            )
prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
retriever = db.as_retriever()

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_generated_question=False,
    rephrase_question=True,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt}
    )

def chatllama(user_msg, history):
    result = chain(user_msg, history)
    return result['answer']

def generate_response(input_text):
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 1024})
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    generate_response(text)
