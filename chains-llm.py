from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import os

DB_FAISS_PATH = 'vs-db/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db, pdf_file):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={
                                               'k': 3,
                                               'filter': {'source': pdf_file}
                                           }),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# Loading the model
def load_llm():
    llm = CTransformers(
        model="mistral-7b-instruct-v0.1.Q8_0.gguf",
        model_type="mistral",
        max_new_tokens=256,
        temperature=0.2 ,
        context_length = 2048,
        gpu_layers = 32 ,# 32 to put all mistral layers on GPU, might differ for other models
        threads = -1
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    pdf_files = os.listdir("data/")
    qa_with_filenames = []
    for pdf_file in pdf_files:
        qa_chain = retrieval_qa_chain(llm, qa_prompt, db, pdf_file)
        qa_with_filenames.append((qa_chain, pdf_file))
    return qa_with_filenames

# Chainlit code
@cl.on_chat_start
async def start():
    chains = qa_bot()
    cl.user_session.set("chains", chains)
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Bot. What is your query?"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    chains = cl.user_session.get("chains")
    responses = []

    for qa_chain, pdf_file in chains:
        res = await qa_chain.acall(message.content)
        answer = res["result"]
        sources = pdf_file

        if sources:
            answer += f"\nSources: {sources}"


        responses.append(answer)

    final_response = "\n\n".join(responses)
    await cl.Message(content=final_response).send()

