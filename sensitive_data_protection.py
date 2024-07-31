import gradio as gr
import os
from langchain.schema import Document
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from operator import itemgetter
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough

# Sample document content
document_content = """Date: October 19, 2021
 Witness: John Doe
 Subject: Testimony Regarding the Loss of Wallet
 Testimony Content:
 Hello Officer,
 My name is John Doe and on October 19, 2021, my wallet was stolen in the vicinity of Kilmarnock during a bike trip. This wallet contains some very important things to me.
 Firstly, the wallet contains my credit card with number 4111 1111 1111 1111, which is registered under my name and linked to my bank account, PL61109010140000071219812874.
 Additionally, the wallet had a driver's license - DL No: 999000680 issued to my name. It also houses my Social Security Number, 602-76-4532.
 """

# Function to set up the environment and perform anonymization
def setup_anonymizer(api_key, document_content):
    os.environ['OPENAI_API_KEY'] = api_key

    # Initialize the anonymizer
    anonymizer = PresidioReversibleAnonymizer(add_default_faker_operators=True, faker_seed=42)

    # Create a Document object
    documents = [Document(page_content=document_content)]

    # Anonymize the document content
    for doc in documents:
        doc.page_content = anonymizer.anonymize(doc.page_content)

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Index the chunks
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(chunks, embeddings)
    retriever = docsearch.as_retriever()

    # Create the anonymizer chain
    template = """Answer the question based only on the following context:
    {context}
    Question: {anonymized_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0.3)

    _inputs = RunnableParallel(
        question=RunnablePassthrough(),
        anonymized_question=RunnableLambda(anonymizer.anonymize),
    )

    anonymizer_chain = (
        _inputs
        | {
            "context": itemgetter("anonymized_question") | retriever,
            "anonymized_question": itemgetter("anonymized_question"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    chain_with_deanonymization = anonymizer_chain | RunnableLambda(anonymizer.deanonymize)

    return documents[0].page_content, anonymizer_chain, chain_with_deanonymization

# Function to get the answer to a question
def get_answer(question, anonymizer_chain, chain_with_deanonymization):
    anonymized_answer = anonymizer_chain.invoke(question)
    deanonymized_answer = chain_with_deanonymization.invoke(question)
    return anonymized_answer, deanonymized_answer

# Gradio interface for document anonymization
def anonymize_document(api_key, document_content):
    anonymized_content, anonymizer_chain, chain_with_deanonymization = setup_anonymizer(api_key, document_content)
    return anonymized_content, anonymizer_chain, chain_with_deanonymization

# Gradio interface for question answering
def answer_question(question, anonymizer_chain, chain_with_deanonymization):
    anonymized_answer, deanonymized_answer = get_answer(question, anonymizer_chain, chain_with_deanonymization)
    return anonymized_answer, deanonymized_answer

# Create Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# RAG with Sensitive-Data-Protection")

    with gr.Tab("Anonymize Document"):
        with gr.Row():
            api_key_input = gr.Textbox(lines=1, placeholder="Enter your OpenAI API Key", label="OpenAI API Key", type='password')
            document_content_input = gr.Textbox(lines=10, placeholder="Enter document content", label="Document Content", value=document_content)
            anonymize_button = gr.Button("Anonymize")

        anonymized_content_output = gr.Textbox(lines=10, label="Anonymized Document Content")
        anonymizer_chain_state = gr.State()
        chain_with_deanonymization_state = gr.State()

        anonymize_button.click(
            fn=anonymize_document,
            inputs=[api_key_input, document_content_input],
            outputs=[anonymized_content_output, anonymizer_chain_state, chain_with_deanonymization_state]
        )

    with gr.Tab("Answer Questions"):
        with gr.Row():
            question_input = gr.Textbox(lines=1, placeholder="Enter your question", label="Question")
            answer_button = gr.Button("Get Answer")

        anonymized_answer_output = gr.Textbox(lines=2, label="Anonymized Answer")
        deanonymized_answer_output = gr.Textbox(lines=2, label="Deanonymized Answer")

        answer_button.click(
            fn=answer_question,
            inputs=[question_input, anonymizer_chain_state, chain_with_deanonymization_state],
            outputs=[anonymized_answer_output, deanonymized_answer_output]
        )

demo.launch()