import logging
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from state import State

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def initialize_llm():
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    chat_model = os.environ["AZURE_OPENAI_MODEL"]
    logger.info(
        f"Initializing LLM with endpoint: {azure_endpoint}, deployment: {azure_deployment}, api_version: {openai_api_version}, model: {chat_model}"
    )
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        openai_api_version=openai_api_version,
        model=chat_model,
    )

    embeddings_model = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL"]
    embeddings_deployment = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
    embeddings_api_version = os.environ["AZURE_OPENAI_EMBEDDINGS_API_VERSION"]
    logger.info(
        f"Initializing Azure Embeddings with model: {embeddings_model}, deployment: {embeddings_deployment}, api_version: {embeddings_api_version}"
    )
    embeddings = AzureOpenAIEmbeddings(
        model=embeddings_model,
        azure_deployment=embeddings_deployment,
        openai_api_version=embeddings_api_version,
    )

    return llm, embeddings


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)

    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(
        f"Source: {doc.metadata['source'].split('/')[1] if '/' in doc.metadata['source'] else doc.metadata['source']}, "
        f"Page: {doc.metadata['page']}\n Content: {doc.page_content}"
        for doc in state["context"]
    )
    template = f"""You are an assistant to help solve questions about the {os.environ["BOARD_GAME"]} board game.
        Use the following retrieved context fragments to answer the question.
        If you don't know the answer, simply say you don't know.
        Your goal is to answer the question as best as possible.
        
        Your response must follow this format:
        ```
        <your answer here>

        Docs:
        <Source1>: <Page1>
        <Source2>: <Page2>
        ```
        
        Question: {{question}}
        Context: {{context}}
        Answer:"""
    prompt = PromptTemplate.from_template(template)

    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)

    return {"answer": response.content}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"I'm a {os.environ['BOARD_GAME']} bot! Please ask me anything from the game and I'll do my best to help you.",
    )


async def question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    logger.info(
        f"Received question: {question} from chat_id: {update.effective_chat.id}"
    )
    response = graph.invoke({"question": question})
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=response["answer"]
    )


def build_vector_db():
    database = os.environ["QDRANT_DATABASE"]
    client = QdrantClient(path=f"databases/{database}", permission="rw")

    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    collection_name = f"{database}_collection"
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=azure_embeddings
    )

    if os.environ["LOAD_DOCS"] == "False":
        return vector_store
    for root, _, files in os.walk("docs/"):
        logger.info(f"Loading documents from {root}...")
        for filename in files:
            pages = []
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, filename), extract_images=False)
                for page in loader.load():
                    pages.append(page)
                vector_store.add_documents(pages)
                logger.info(f"Loaded document {filename} from {root}")
        logger.info(f"Loaded {len(files)} documents from {root}")

    logger.info("Loaded all documents")

    return vector_store


def generate_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph


def build_telegram_bot():
    application = ApplicationBuilder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    start_handler = CommandHandler("start", start)
    question_handler = MessageHandler(filters.TEXT & filters.ChatType.PRIVATE, question)
    mention_handler = MessageHandler(
        filters.Entity("mention") & filters.ChatType.GROUP, question
    )
    application.add_handler(start_handler)
    application.add_handler(question_handler)
    application.add_handler(mention_handler)
    application.run_polling()


if __name__ == "__main__":
    llm, azure_embeddings = initialize_llm()
    vector_store = build_vector_db()
    graph = generate_graph()
    build_telegram_bot()
