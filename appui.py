import streamlit as st
import requests

# 配置你的 FastAPI 后端地址
API_BASE_URL = "http://127.0.0.1:8000"

# --- 网页基础设置 ---
st.set_page_config(page_title="Gallery AI 助手", page_icon="🤖", layout="centered")
st.title("🤖 Gallery AI 智能聊天机器人")
st.caption("基于 RAG 技术的私有知识库系统 (支持 PDF 与网页链接)")

# --- 初始化聊天记录记忆 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的 AI 助手。请在左侧上传 PDF 或输入网页链接让我学习，然后向我提问吧！"}
    ]

# --- 左侧边栏：知识库管理 ---
with st.sidebar:
    st.header("📂 知识库管理")
    
    # 模块 1：上传 PDF 功能
    st.subheader("📄 上传文档 (PDF)")
    uploaded_file = st.file_uploader("选择一个 PDF 文件", type=["pdf"], label_visibility="collapsed")
    if st.button("📤 学习 PDF"):
        if uploaded_file is not None:
            with st.spinner("AI 正在拼命阅读中..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_BASE_URL}/admin/ingest-pdf", files=files)
                    if res.status_code == 200:
                        st.success(f"成功学习文档: {uploaded_file.name}")
                    else:
                        st.error("上传失败，请检查后端报错。")
                except Exception as e:
                    st.error(f"连接后端失败: {e}")
        else:
            st.warning("请先选择一个文件！")
            
    st.divider() # 这是一条华丽的分割线
    
    # 模块 2：抓取网页 URL 功能 (👈 这是为你新加的！)
    st.subheader("🌐 抓取网页 (URL)")
    input_url = st.text_input("输入网址 (以 http 或 https 开头)", placeholder="https://example.com")
    if st.button("🔍 学习网页"):
        if input_url:
            with st.spinner("AI 正在上网冲浪抓取网页..."):
                try:
                    # 注意这里是用 json 发送数据，对应你的 UrlIngestRequest 模型
                    res = requests.post(f"{API_BASE_URL}/admin/ingest-url", json={"url": input_url})
                    if res.status_code == 200:
                        st.success(f"成功学习网页内容！")
                    else:
                        st.error("抓取失败，可能是该网站防爬虫，或后端报错。")
                except Exception as e:
                    st.error(f"连接后端失败: {e}")
        else:
            st.warning("请先输入一个网址！")

# --- 主界面：聊天区域 ---
# 1. 渲染历史聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 接收用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户的提问
    with st.chat_message("user"):
        st.markdown(prompt)
    # 把用户提问加入记忆
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. 向你的后端发送请求
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            try:
                response = requests.post(f"{API_BASE_URL}/query", json={"question": prompt})
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "抱歉，我没有得出答案。")
                    topic = data.get("topic", "Unknown")
                    sources = data.get("sources", [])
                    
                    # 拼装漂亮的回答卡片
                    source_str = "\n".join([f"- `{s}`" for s in sources]) if sources else "无"
                    reply_text = f"""
{answer}

---
**🏷️ 话题分类：** `{topic}`  
**📚 知识来源：**  
{source_str}
"""
                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                else:
                    st.error("AI 引擎开小差了，请稍后再试。")
            except Exception as e:
                st.error(f"无法连接到 AI 大脑，请确认黑窗口里的 main.py 已启动。({e})")