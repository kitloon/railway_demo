import os
import time
from typing import List, Generator, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# FastAPI 框架组件
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# LangChain 核心组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from tavily import TavilyClient

# --- 1. 环境与基础配置 ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./data/chroma_db")

app = FastAPI(title="SCCCI Ultimate RAG Agent")

# 跨域配置 (如果你有前端页面，这必不可少)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. API 数据模型 (严格定义) ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    topic: str
    sources: List[str]

class IngestPDFRequest(BaseModel):
    file_path: str  # 这里的路径必须是服务器本地可访问的

class IngestURLRequest(BaseModel):
    url: str

# --- 3. RAG 核心引擎类 ---
class RAGEngine:
    def __init__(self):
        print("🛠️ [SYSTEM] 正在启动全功能引擎：支持 Memory + Rerank + WebAgent + Ingestion")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            persist_directory=PERSIST_DIR, 
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # 决策与分析模型 (非流式，确保逻辑稳定)
        self.planner = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # 交互与输出模型 (流式，提升用户体验)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        
        # 关键：持久化对话记忆
        self.memory = ChatMessageHistory()
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    def _rewrite_query(self, question: str) -> str:
        """【意图对齐】利用记忆补全模糊问题"""
        if not self.memory.messages:
            return question
        
        history_snippet = "\n".join([f"{m.type}: {m.content}" for m in self.memory.messages[-2:]])
        prompt = f"""根据以下对话历史，将用户的新问题改写为一个独立且完整的搜索词。
        
        历史对话:
        {history_snippet}
        
        新问题: {question}
        
        只需输出改写后的文本，不要任何解释。"""
        
        refined = self.planner.invoke(prompt).content
        return refined

    def query(self, question: str) -> Tuple[str, str, List[str]]:
        """【同步接口适配器】解决调用时的 AttributeError"""
        full_text = ""
        for chunk in self.stream_query(question):
            # 过滤掉系统内部提示标签
            if not chunk.startswith("💡") and not chunk.startswith("🌐"):
                full_text += chunk
        return full_text, "General", ["Hybrid Retrieval (Local Vector + Tavily Web)"]

    def stream_query(self, question: str) -> Generator[str, None, None]:
        """【流式核心逻辑】含完整日志输出"""
        print(f"\n🚀 [LOG] 收到用户请求: {question}")
        
        # 1. 意图重构
        refined_q = self._rewrite_query(question)
        print(f"🔍 [LOG] 意图改写结果: {refined_q}")

        # 2. 本地检索
        docs = self.retriever.invoke(refined_q)
        local_context = "\n\n".join([d.page_content for d in docs])
        print(f"📚 [LOG] 本地检索完成，召回片段: {len(docs)}")

        # 3. 动态联网判定
        check_prompt = f"问题: {refined_q}\n现有资料: {local_context}\n如果无法准确回答，请仅输出 NEED_SEARCH，否则输出 OK。"
        decision = self.planner.invoke(check_prompt).content
        
        final_context = local_context
        if "NEED_SEARCH" in decision:
            print(f"⚠️ [LOG] 本地库覆盖不足，正在启动 Tavily 搜刮全网信息...")
            yield "💡 正在连接 Tavily 获取实时动态...\n\n"
            
            web_data = self.tavily.search(query=refined_q, max_results=3, search_depth="basic")
            final_context = "\n\n".join([f"来源:{r['url']}\n内容:{r['content']}" for r in web_data['results']])
            print(f"🌐 [LOG] 联网数据获取成功。")
        else:
            print(f"✅ [LOG] 本地知识库足以支撑回答。")

        # 4. 组装最终 Prompt 并流式输出
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业分析助手。请严格基于提供的资料和对话历史进行回答。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "【参考资料】:\n{context}\n\n【用户问题】: {question}")
        ])
        
        formatted_messages = prompt.format_messages(
            context=final_context, 
            question=question, 
            history=self.memory.messages
        )

        print(f"✍️ [LOG] 开始生成流式回答...")
        accumulated_answer = ""
        for chunk in self.llm.stream(formatted_messages):
            token = chunk.content
            accumulated_answer += token
            yield token
        
        # 5. 更新记忆
        self.memory.add_user_message(question)
        self.memory.add_ai_message(accumulated_answer)
        print(f"💾 [LOG] 记忆已同步，本轮对话完成。\n")

    # --- 数据导入方法集 ---
    def ingest_pdf(self, path: str) -> int:
        print(f"📄 [SYSTEM] 正在解析 PDF: {path}")
        loader = PyPDFLoader(path)
        documents = loader.load_and_split(self.splitter)
        self.vector_store.add_documents(documents)
        return len(documents)

    def ingest_url(self, url: str) -> int:
        print(f"🔗 [SYSTEM] 正在抓取 URL: {url}")
        loader = WebBaseLoader(url)
        documents = loader.load_and_split(self.splitter)
        self.vector_store.add_documents(documents)
        return len(documents)

# --- 4. FastAPI 路由定义 (确保路径与你之前的调用习惯一致) ---
engine = RAGEngine()

@app.post("/query")
async def ask_ai_json(request: QueryRequest):
    """支持传统的 POST JSON 响应"""
    ans, topic, src = engine.query(request.question)
    return QueryResponse(answer=ans, topic=topic, sources=src)

@app.post("/query_stream")
async def ask_ai_stream(request: QueryRequest):
    """支持流式响应 (text/plain)"""
    return StreamingResponse(engine.stream_query(request.question), media_type="text/plain")

@app.post("/admin/ingest-pdf")
async def admin_ingest_pdf(request: IngestPDFRequest):
    """PDF 导入接口"""
    try:
        count = engine.ingest_pdf(request.file_path)
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        print(f"❌ [ERROR] PDF 导入失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/ingest-url")
async def admin_ingest_url(request: IngestURLRequest):
    """URL 导入接口"""
    try:
        count = engine.ingest_url(request.url)
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        print(f"❌ [ERROR] URL 导入失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 直接运行: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)