
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from typing import Annotated
from newsapi import NewsApiClient
from langchain.tools import Tool
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import hashlib

def check_auth():
    """Password protection using .env"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔒 Login")
        password = st.text_input("Enter password:", type="password", key="pwd_input")
        if st.button("Login"):
            if hashlib.sha256(password.encode()).hexdigest() == os.getenv("PASSWORD"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

check_auth()  # Blocks unauthenticated access

def newsapi_search(query: str, top_k: int = 3) -> str:
    """Search news using NewsAPI"""
    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
    try:
        results = newsapi.get_everything(q=query, language='en', page_size=top_k)
        if not results['articles']:
            return "No news found on this topic."
        return "\n\n".join(
            f"**{article['title']}**\n"
            f"Source: {article['source']['name']}\n"
            f"Published: {article['publishedAt'][:10]}\n"
            f"{article['description']}\n"
            f"[Read more]({article['url']})"
            for article in results['articles']
        )
    except Exception as e:
        return f"NewsAPI error: {str(e)}"

# Create LangChain Tool
news_tool = Tool.from_function(
    func=newsapi_search,
    name="NewsAPI",
    description="Useful for finding current news articles. Input should be a search query about recent news."
)

# --- Tool setup (from your notebook) ---
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arxiv papers")

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

tavily = TavilySearchResults()

tools = [arxiv, wiki, tavily,news_tool]

# --- Google Sheets History ---
def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", 
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        os.getenv("GOOGLE_SHEETS_CREDS"), scope)
    return gspread.authorize(creds)

def save_to_sheet(role: str, content: str, tool: str = None):
    try:
        client = get_gspread_client()
        sheet = client.open(os.getenv("SHEET_NAME")).sheet1
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            role,
            content,
            tool if tool else "N/A"
        ])
    except Exception as e:
        st.error(f"Failed to save history: {e}")


# --- LLM setup (from your notebook) ---
llm = ChatGroq(model="qwen-qwq-32b")
llm_with_tools = llm.bind_tools(tools=tools)

# --- LangGraph state and graph setup ---
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()
# --- End of backend setup ---

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-Tool AI Chatbot", page_icon="🤖")
st.title("🤖 Multi-Tool AI Chatbot")
st.caption("Ask about AI news, research papers, Wikipedia topics, and more!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    try:  # Load history from sheet
        client = get_gspread_client()
        sheet = client.open(os.getenv("SHEET_NAME")).sheet1
        for row in sheet.get_all_records()[::-1]:  # Show newest first
            st.session_state.messages.append({
                "role": row["Role"],
                "content": row["Content"],
                "tool": row.get("Tool")
            })
    except:
        pass

# Display chat history
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])
        

# User input box
if prompt := st.chat_input("Ask anything about research, news, or general knowledge..."):
    # Save user message
    st.session_state.messages.append({"role": "human", "content": prompt})
    save_to_sheet("human", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)    
    
    # Get AI response
    response = graph.invoke({"messages": st.session_state.messages})
    new_messages = response["messages"][len(st.session_state.messages):]
    
    # Process and display
    for msg in new_messages:
        content = getattr(msg, "content", str(msg))
    
    # Safer tool detection
    tool_used = None
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        try:
            tool_used = msg.tool_calls[0].get('tool_name') or getattr(msg.tool_calls[0], 'tool_name', None)
        except (AttributeError, IndexError, TypeError):
            tool_used = None
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "tool": tool_used
    })
    
    # Save to Google Sheets
    if tool_used:
        save_to_sheet("assistant", content, tool_used)
    else:
        save_to_sheet("assistant", content, "direct_response")
    
    with st.chat_message("assistant"):
        st.markdown(content)
     
