import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 自定义模块导入
from MySparkAIEmbeddings import MySparkAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms.sparkllm import SparkLLM

# 导入 chromadb 官方 client
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

_ = load_dotenv(find_dotenv())  # 加载 .env 文件

# 获取环境变量 API_KEY
IFLYTEK_SPARK_APP_ID = os.environ["IFLYTEK_SPARK_APP_ID"]
IFLYTEK_SPARK_API_KEY = os.environ["IFLYTEK_SPARK_API_KEY"]
IFLYTEK_SPARK_API_SECRET = os.environ["IFLYTEK_SPARK_API_SECRET"]


# 自定义 EmbeddingFunction 适配器
class LangChainEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, texts: Documents) -> Embeddings:
        return [self.embedding_model.embed_query(text) for text in texts]


# 2、定义get_retriever函数，该函数返回一个检索器
def get_retriever():
    # 定义 Embeddings
    embedding = MySparkAIEmbeddings()
    adapter = LangChainEmbeddingAdapter(embedding)

    # 连接到远程 chromadb-server（你需要自己部署）
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        settings=chromadb.config.Settings(allow_reset=True)
    )

    # 获取或创建 collection
    collection = client.get_or_create_collection(
        name="langchain",
        embedding_function=adapter
    )

    # 返回一个模拟 retriever 的函数
    def retriever(query):
        embedded_query = embedding.embed_query(query)
        results = collection.query(
            query_embeddings=[embedded_query],
            n_results=4
        )
        # 构造类似 Document 的结构
        from langchain_core.documents import Document
        docs = [
            Document(page_content=doc, metadata={})
            for doc in results["documents"][0]
        ]
        return docs

    return retriever


# 3、定义combine_docs函数， 该函数处理检索器返回的文本
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])



# 4、定义get_qa_history_chain函数，该函数可以返回一个检索问答链
def get_qa_history_chain():
    retriever = get_retriever()

    llm = SparkLLM(
        model="Spark Max",
        app_id=IFLYTEK_SPARK_APP_ID,
        api_key=IFLYTEK_SPARK_API_KEY,
        api_secret=IFLYTEK_SPARK_API_SECRET,
        spark_api_url="wss://spark-api.xf-yun.com/v3.5/chat",
        spark_llm_domain="generalv3.5",
        request_timeout=60
    )

    condense_question_system_prompt = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), lambda x: x["input"] | retriever),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context=retrieve_docs,
    ).assign(answer=qa_chain)

    return qa_history_chain


# 5、定义gen_response函数，它接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


# 6、定义main函数，该函数制定显示效果与逻辑
def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()

    messages = st.container(height=550)

    for message in st.session_state.messages:
        with messages.chat_message(message[0]):
            st.write(message[1])

    if prompt := st.chat_input("Say something"):
        st.session_state.messages.append(("human", prompt))

        with messages.chat_message("human"):
            st.write(prompt)

        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )

        with messages.chat_message("ai"):
            output = st.write_stream(answer)

        st.session_state.messages.append(("ai", output))
        print(f"AI 回答: {output}")


if __name__ == "__main__":
    main()
