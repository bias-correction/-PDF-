import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
                            return_messages=True,
                            memory_key='chat_history',
                            output_key='answer')
    st.session_state["messages"] = []

st.title("📑 AI智能PDF问答工具")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")

uploaded_file = st.file_uploader("上传你的PDF文件：", type=['pdf'])
question = st.text_input("对PDF的内容进行提问", disabled=not uploaded_file)
submit = st.button("提交")

if submit:
    if not openai_api_key:
        st.info("请输入你的OpenAI API Key")
        st.stop()
    if not question:
        st.info("请输入提问的问题")
        st.stop()

    with st.spinner("AI正在思考中，请稍等..."):
        result = qa_agent(openai_api_key,
                      st.session_state["memory"],
                      uploaded_file, question)
        st.write("### 答案")
        st.write(result['answer'])
        st.session_state["messages"].append({"question": question, "answer": result['answer']})

        with st.expander("历史消息"):
            for (index, msg) in enumerate(st.session_state["messages"]):
                if index > 0:
                    st.divider()
                st.chat_message("human").write(msg["question"])
                st.chat_message("ai").write(msg["answer"])

