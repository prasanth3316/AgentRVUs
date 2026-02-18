import streamlit as st
import requests
import os
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

# --- 1. CONFIGURATION ---
AKASH_API_KEY = "akml-NSDnxHaUEDUafkHmpKcWBGUljRiKmgNT" 
AKASH_BASE_URL = st.secrets["AKASH_API_KEY"]
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

YEAR_DATASETS = {
    "2026": "469d7c36-73d6-51f5-9bb2-f0d3e2cd54c5",
    "2025": "eaaa3c55-770e-5e77-8f1d-615e46c1a789",
    "2024": "1e6deb9e-153e-5a50-a9ef-d41c057420c2",
    "2023": "1a9bd6e8-10cb-59e6-9adc-c079c248b378",
    "2022": "3d703d65-fcc5-5d40-ba65-1b57a5a8c517",
    "2021": "8c3a7088-6aab-5c89-b41a-b3da645288a7",
    "2018": "36fcf45a-1ed2-5349-adf6-3a97afe8d958"
}

# --- 2. TOOL DEFINITION ---
def fetch_single_rvu(tool_input: str):
    """Validates ONE code for ONE year. Format: 'CODE, YEAR'"""
    url = "https://pfs.data.cms.gov/api/1/datastore/sql"
    try:
        # Clean input and handle cases where LLM sends just the code
        clean_input = tool_input.replace('"', '').replace("'", "").strip()
        parts = [p.strip().upper() for p in clean_input.split(",")]
        
        hcpc = parts[0]
        year = parts[1] if len(parts) > 1 else "2026"

        dist_id = YEAR_DATASETS.get(year, YEAR_DATASETS["2026"])
        query = f'[SELECT hcpc,rvu_work FROM {dist_id}][WHERE hcpc = "{hcpc}"][LIMIT 1]'
        
        r = requests.get(url, params={"query": query}, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = data if isinstance(data, list) else data.get('results', [])

        if results:
            res = results[0]
            return f"DATA: HCPC {res['hcpc']} in {year} has a Work RVU of {res['rvu_work']}."
        return f"NOT FOUND: Code {hcpc} not found in {year} dataset."
    except Exception as e:
        return f"ERROR: Database lookup failed. {str(e)}"

# --- 3. UPDATED AGENT SETUP (Conversational Logic) ---
llm = ChatOpenAI(
    openai_api_key=AKASH_API_KEY,
    base_url=AKASH_BASE_URL,
    model_name=MODEL_NAME,
    temperature=0 
)

tools = [
    Tool(
        name="CMS_Single_Lookup",
        func=fetch_single_rvu,
        description="Lookup Work RVU for a specific code and year. Input must be 'CODE, YEAR'."
    )
]

# --- 3. UPDATED AGENT SETUP (Anti-Loop Version) ---
template = """You are a professional Medicare Coding & RVU Assistant.
You have access to CMS data for years: 2018, 2021, 2022, 2023, 2024, 2025, and 2026.
If a user asks for a year outside this range, inform them it is not available.
PROCEDURE:
0. If general question, answer directly. If specific code/year question, follow steps below.
1. Check CHAT HISTORY for previously discussed codes/years.
2. For each NEW code requested, call CMS_Single_Lookup exactly ONCE.
3. If you see an "Observation" that starts with "DATA:", you have the information. 
4. DO NOT repeat a tool call for the same code and year.
5. Once you have all data points, immediately provide the Final Answer.

TOOLS:
------
{tools}

To use a tool, please use the following format:
Thought: I need to look up HCPC [CODE] for [YEAR].
Action: [{tool_names}]
Action Input: CODE, YEAR
Observation: [result]

... (Once all codes are looked up)
Thought: I have retrieved all necessary RVU data. I will now summarize.
Final Answer: [A clear breakdown of the codes, their individual RVUs, and the total sum if applicable]

CHAT HISTORY:
{chat_history}

USER INPUT:
{input}

{agent_scratchpad}"""

# --- 4. STREAMLIT INTERFACE ---
st.set_page_config(page_title="Akash RVU Assistant", page_icon="🏥", layout="wide")

# Sidebar for Session Controls
with st.sidebar:
    st.title("Settings")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.rerun()

st.title("🏥 Medicare RVU Conversational Pro")
st.caption("Maintains context across multiple questions (2021-2026)")

# Initialize Memory and Messages
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Build Agent
prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)

# Ensure handle_parsing_errors is True to help the LLM recover from minor format slips
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=st.session_state.memory, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15 # Lowering this slightly helps catch loops faster
)

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if user_input := st.chat_input("Ask about codes, RVUs, or specific years..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # The agent_executor automatically pulls from st.session_state.memory
                response = agent_executor.invoke({"input": user_input})
                output = response["output"]
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
