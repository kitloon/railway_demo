import os
from typing import List, Generator, Tuple
from dotenv import load_dotenv

# LangChain Core Components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 1. Environment and Basic Configuration ---
load_dotenv()
PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./data/chroma_db")


# --- 2. RAG Core Engine Class ---
class RAGEngine:
    def __init__(self):
        print("🛠️ [SYSTEM] Starting Engine: Memory + MMR Rerank + Source Tracking enabled")
        
        # 🟢 【新增调试代码】打印出当前容器能看到的所有变量名，抓出内鬼
        print("📋 [DEBUG] 容器内所有可用的环境变量列表（Keys）:")
        print(list(os.environ.keys()))
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        # 🟢 【新增调试代码】顺便看看是不是名字带了空格
        if api_key:
            print(f"🔍 [DEBUG] 成功获取到 Key，长度为: {len(api_key)}，开头前4位: {api_key[:4]}")
        else:
            print("❌ [DEBUG] os.getenv('OPENAI_API_KEY') 返回了 None！")

        if not api_key:
            raise ValueError("❌ 错误: 未能在环境变量中找到 OPENAI_API_KEY。请检查上方打印的列表里有没有它。")

        # 显式传入 api_key
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        self.vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings
        )

        # MMR (Maximal Marginal Relevance) reduces redundant chunks and improves diversity
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6}
        )

        # Planner: used for query rewriting and topic classification (non-streaming)
        self.planner = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        # Main LLM: used for final answer generation
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True, openai_api_key=api_key)

        # Persistent conversation memory
        self.memory = ChatMessageHistory()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # ------------------------------------------------------------------
    # Query Rewriting
    # ------------------------------------------------------------------
    def _rewrite_query(self, question: str) -> str:
        """Rewrite ambiguous questions using recent conversation history."""
        if not self.memory.messages:
            return question

        history_snippet = "\n".join(
            [f"{m.type}: {m.content}" for m in self.memory.messages[-4:]]
        )
        # Prompt translated to English for logic consistency
        prompt = (
            f"Based on the following conversation history, rewrite the user's new question into a standalone and complete search term.\n\n"
            f"Conversation History:\n{history_snippet}\n\n"
            f"New Question: {question}\n\n"
            f"Output only the rewritten text without any explanation."
        )
        refined = self.planner.invoke(prompt).content.strip()
        print(f"🔍 [LOG] Intent Rewriting Result: {refined}")
        return refined

    # ------------------------------------------------------------------
    # Topic Classification
    # ------------------------------------------------------------------
    def _classify_topic(self, question: str, context: str) -> str:
        """Classify the topic of the answer based on question and retrieved context."""
        prompt = (
            f"Based on the question and references, summarize the topic in 2–4 English words (e.g., Product Info, Tech Support, Pricing, History).\n\n"
            f"Question: {question}\n"
            f"Context Summary: {context[:500]}\n\n"
            f"Output only the topic category words."
        )
        topic = self.planner.invoke(prompt).content.strip()
        return topic if topic else "General"

    # ------------------------------------------------------------------
    # Source Extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_sources(docs) -> List[str]:
        """Extract unique source filenames/URLs from retrieved documents."""
        seen = set()
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    # ------------------------------------------------------------------
    # Main Query Interface (sync, used by FastAPI /query)
    # ------------------------------------------------------------------
    def query(self, question: str) -> Tuple[str, str, List[str]]:
        """Synchronous query: returns (answer, topic, sources)."""
        print(f"\n🚀 [LOG] Received request: {question}")

        # 1. Query rewriting
        refined_q = self._rewrite_query(question)

        # 2. Local vector retrieval (MMR)
        docs = self.retriever.invoke(refined_q)
        print(f"📚 [LOG] Retrieval complete, chunks recalled: {len(docs)}")

        # 3. If nothing retrieved, respond honestly
        if not docs:
            answer = "I'm sorry, I couldn't find any information related to this question in the knowledge base. Please upload relevant PDFs or URLs first."
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)
            return answer, "No relevant data", []

        local_context = "\n\n".join([d.page_content for d in docs])
        sources = self._extract_sources(docs)

        # 4. Confidence check
        check_prompt = (
            f"You are a strict knowledge base assistant. Here is the retrieved context:\n{local_context}\n\n"
            f"User Question: {question}\n\n"
            f"Does the context provide enough information to answer the question? Answer only YES or NO."
        )
        decision = self.planner.invoke(check_prompt).content.strip().upper()

        if "NO" in decision:
            answer = (
                "I'm sorry, I cannot accurately answer this question based on the current knowledge base. "
                "Please try uploading more relevant materials."
            )
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)
            return answer, "Insufficient Coverage", sources

        # 5. Build prompt and generate answer
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional knowledge base assistant. "
                "Strictly use the provided [Context] to answer the question. "
                "If the information is not in the context, say 'Information not mentioned'. "
                "Be concise and accurate. Use English for the response."
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "[Context]:\n{context}\n\n[User Question]: {question}")
        ])

        formatted_messages = prompt.format_messages(
            context=local_context,
            question=question,
            history=self.memory.messages
        )

        print("✍️ [LOG] Generating answer...")
        answer_chunks = []
        for chunk in self.llm.stream(formatted_messages):
            answer_chunks.append(chunk.content)
        answer = "".join(answer_chunks)

        # 6. Classify topic
        topic = self._classify_topic(question, local_context)

        # 7. Update memory
        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)
        print(f"💾 [LOG] Memory synced. Session complete. Topic: {topic}\n")

        return answer, topic, sources

    # ------------------------------------------------------------------
    # Streaming Query Interface (used by /query_stream)
    # ------------------------------------------------------------------
    def stream_query(self, question: str) -> Generator[str, None, None]:
        """Streaming query: yields answer tokens one by one."""
        print(f"\n🚀 [LOG] Received streaming request: {question}")

        refined_q = self._rewrite_query(question)
        docs = self.retriever.invoke(refined_q)
        print(f"📚 [LOG] Retrieval complete, chunks recalled: {len(docs)}")

        if not docs:
            msg = "I'm sorry, I couldn't find any information related to this question in the knowledge base."
            self.memory.add_user_message(question)
            self.memory.add_ai_message(msg)
            yield msg
            return

        local_context = "\n\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional knowledge base assistant. "
                "Strictly use the provided [Context] to answer. "
                "If not found, say 'Information not mentioned'. Use English."
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "[Context]:\n{context}\n\n[User Question]: {question}")
        ])

        formatted_messages = prompt.format_messages(
            context=local_context,
            question=question,
            history=self.memory.messages
        )

        accumulated_answer = ""
        for chunk in self.llm.stream(formatted_messages):
            token = chunk.content
            accumulated_answer += token
            yield token

        self.memory.add_user_message(question)
        self.memory.add_ai_message(accumulated_answer)
        print("💾 [LOG] Streaming memory synced.\n")

    # ------------------------------------------------------------------
    # Knowledge Base Management
    # ------------------------------------------------------------------
    def ingest_pdf(self, path: str) -> int:
        """Load a PDF, split into chunks, and add to the vector store."""
        print(f"📄 [SYSTEM] Parsing PDF: {path}")
        loader = PyPDFLoader(path)
        documents = loader.load_and_split(self.splitter)
        # Normalize source metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(path)
        self.vector_store.add_documents(documents)
        print(f"✅ [SYSTEM] PDF Ingestion complete. {len(documents)} chunks added.")
        return len(documents)

    def ingest_url(self, url: str) -> int:
        """Scrape a URL, split into chunks, and add to the vector store."""
        print(f"🔗 [SYSTEM] Scraping URL: {url}")
        loader = WebBaseLoader(url)
        documents = loader.load_and_split(self.splitter)
        for doc in documents:
            doc.metadata["source"] = url
        self.vector_store.add_documents(documents)
        print(f"✅ [SYSTEM] URL Ingestion complete. {len(documents)} chunks added.")
        return len(documents)

    def list_sources(self) -> List[str]:
        """Return a deduplicated list of all ingested sources."""
        try:
            all_docs = self.vector_store.get()
            metadatas = all_docs.get("metadatas", [])
            sources = list({m.get("source", "Unknown") for m in metadatas if m})
            return sorted(sources)
        except Exception as e:
            print(f"❌ [ERROR] Failed to list sources: {e}")
            return []

    def delete_source(self, source_path: str) -> Tuple[bool, str]:
        """Delete all chunks belonging to a specific source from the vector store."""
        try:
            all_docs = self.vector_store.get()
            ids_to_delete = [
                doc_id
                for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"])
                if meta.get("source") == source_path
            ]
            if not ids_to_delete:
                return False, f"Source not found: {source_path}"
            self.vector_store.delete(ids=ids_to_delete)
            print(f"🗑️ [SYSTEM] Deleted source '{source_path}', removed {len(ids_to_delete)} chunks.")
            return True, f"Successfully deleted {len(ids_to_delete)} chunks (Source: {source_path})"
        except Exception as e:
            print(f"❌ [ERROR] Deletion failed: {e}")
            return False, str(e)
