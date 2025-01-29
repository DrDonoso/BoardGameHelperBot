import os
import logging

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from qdrant_client import QdrantClient
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_community.document_loaders import PyPDFLoader

from state import State

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def initialize_llm():
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"]
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]
    chat_model=os.environ["AZURE_OPENAI_MODEL"]
    logger.info(f"Initializing LLM with endpoint: {azure_endpoint}, deployment: {azure_deployment}, api_version: {openai_api_version}, model: {chat_model}")
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        openai_api_version=openai_api_version,
        model=chat_model
    )

    embeddings_model=os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL"]
    embeddings_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
    embeddings_api_version=os.environ["AZURE_OPENAI_EMBEDDINGS_API_VERSION"]
    logger.info(f"Initializing Azure Embeddings with model: {embeddings_model}, deployment: {embeddings_deployment}, api_version: {embeddings_api_version}")
    embeddings = AzureOpenAIEmbeddings(
        model=embeddings_model,
        azure_deployment=embeddings_deployment,
        openai_api_version=embeddings_api_version
    )
    
    return llm, embeddings

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)

    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}\n{doc.page_content}" for doc in state["context"])
    template = f"""You are an assistant to help solve questions about the {os.environ["BOARD_GAME"]} board game. Use the following retrieved context fragments to answer the question.
        If you don't know the answer, simply say you don't know. Your goal is to answer the question as best as possible. Add as a response in an new line also the doc that is used and the page.
        Question: {{question}}
        Context: {{context}}
        Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)

    return {"answer": response.content}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    response = graph.invoke({"question": question})
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response["answer"])

def build_vector_db():
    client = QdrantClient(path="qdrant-db/")
    
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    collection_name = os.environ["QDRANT_COLLECTION_NAME"]
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=azure_embeddings
    )
    
    if os.environ["LOAD_DOCS"] == "False":
        return vector_store
    for root, _, files in os.walk("docs/"):
        logger.info(f"Loading documents from {root}...")
        for filename in files:
            pages = []
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, filename), extract_images = False)
                for page in loader.load():
                    pages.append(page)
                vector_store.add_documents(pages)
                logger.info(f"Loaded document {filename} from {root}")
        logger.info(f"Loaded {len(files)} documents from {root}")
    
    logger.info(f"Loaded all documents")
    
    return vector_store
    

def generate_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph

def build_telegram_bot():
    application = ApplicationBuilder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    start_handler = CommandHandler('start', start)
    question_handler = MessageHandler(filters.TEXT, question)
    application.add_handler(start_handler)
    application.add_handler(question_handler)
    application.run_polling()

if __name__ == "__main__":
    llm, azure_embeddings = initialize_llm()
    vector_store = build_vector_db()
    graph = generate_graph()
    build_telegram_bot()