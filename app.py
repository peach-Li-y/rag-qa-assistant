import os  # ⚠️ 这行绝对不能删，必须放在最顶部

# ================= 环境配置 =================
# 解决国内下载 HuggingFace 模型网络失败的问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 向量模型已缓存在 ./model_cache；强制离线可避免启动时向 HF 镜像发 HEAD 校验请求而超时卡死。
# 若换到全新机器、本地无缓存需要重新下载，把下面这行改成 '0' 再启动。
os.environ['HF_HUB_OFFLINE'] = '1'

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 加载 .env 文件中的 API Key
load_dotenv()

st.set_page_config(page_title="开源软件知识库问答", page_icon="📚")
st.title("📚 《开源软件构建技术》知识问答助手")
st.caption("基于 LangChain + 豆包大模型 + 本地中文向量模型 搭建的 RAG 应用")

# 1. 检查并加载本地文档
@st.cache_resource(show_spinner="正在解析文档并构建本地知识库（首次运行需下载模型，请稍候）...")
def load_and_process_document():
    pdf_path = "data/《开源软件构建技术》第3章 开源软件版本控制管理-v1.5.pdf"
    
    # ⚠️ 这里调用 os.path.exists，上面必须有 import os
    if not os.path.exists(pdf_path):
        st.error(f"找不到文件：{pdf_path}，请确保它放在 data 文件夹下。")
        return None
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    
    # 向量化：使用本地开源模型
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        cache_folder="./model_cache"
    )
    
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectordb

# 2. 初始化大模型（豆包/火山方舟，标准 OpenAI 兼容通道）
def get_llm():
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "doubao-seed-2-0-mini-260428"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
    )
    return llm

# 3. 主流程
def main():
    vectordb = load_and_process_document()
    
    if vectordb is not None:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        query = st.text_input("请输入关于 Git 或版本控制的问题：")
        
        if query:
            with st.spinner("正在查阅课件并思考..."):
                result = qa_chain.invoke({"query": query})
                
                st.markdown("### 💡 回答：")
                st.write(result["result"])
                
                with st.expander("查看参考来源片段"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**片段 {i+1} (来自第 {doc.metadata.get('page', '?')} 页)**")
                        st.info(doc.page_content[:300] + "...")

if __name__ == "__main__":
    main()