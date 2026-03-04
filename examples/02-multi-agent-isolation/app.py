"""
LangGraph Multi-Vault Example

Two agents, two restricted vaults. Isolation enforced at the infrastructure
layer — not through metadata filtering or prompt rules.

Agents:
- research-agent: authorized to query research-vault only
- atlas-agent:    authorized to query atlas-vault only
- router:         dispatches queries to the appropriate agent (application
                  logic, separate from the isolation mechanism)

Run:
    python app.py

Requires:
    OPENAI_API_KEY environment variable
"""

import os
from pathlib import Path
from typing import Literal, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import subprocess
import time

# =====================================================================
# Config
# =====================================================================

API_URL = "http://127.0.0.1:8000/ctxvault"
VAULTS_DIR = Path(__file__).parent / "vaults"

ATLAS_VAULT = "atlas-vault"
RESEARCH_VAULT = "research-vault"

ATLAS_AGENT_HEADER = {"X-CtxVault-Agent": "atlas-agent"}
RESEARCH_AGENT_HEADER = {"X-CtxVault-Agent": "research-agent"}

# Colors for CLI
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# =====================================================================
# Server
# =====================================================================

def start_server():
    """Start CtxVault API in background."""
    print(f"{BLUE}[SERVER] Starting CtxVault API...{RESET}")
    
    proc = subprocess.Popen(
        ["uvicorn", "ctxvault.api.app:app"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # Wait for server to be ready
    for _ in range(30):
        try:
            requests.get("http://127.0.0.1:8000", timeout=1)
            print(f"{GREEN}[SERVER] API ready{RESET}\n")
            return proc
        except:
            time.sleep(0.3)
    
    raise RuntimeError("CtxVault API failed to start")

# =====================================================================
# CtxVault API helpers
# =====================================================================

def index_vault(vault_name: str):
    """Index documents into vault."""
    requests.put(f"{API_URL}/index",
        json={"vault_name": vault_name}
    )

def query_vault(vault_name: str, query: str, top_k: int = 2):
    """Query vault and return results."""
    response = requests.post(f"{API_URL}/query", 
        headers=ATLAS_AGENT_HEADER if vault_name == ATLAS_VAULT else RESEARCH_AGENT_HEADER,
        json={
        "vault_name": vault_name,
        "query": query,
        "top_k": top_k
    })
    return response.json().get("results", [])

# =====================================================================
# LangGraph State
# =====================================================================

class AgentState(TypedDict):
    query: str
    route: Literal["atlas", "research"]
    context: str
    answer: str

# =====================================================================
# Nodes
# =====================================================================

def router_node(state: AgentState) -> AgentState:
    """Route query to appropriate agent based on content."""
    query = state["query"].lower()
    
    # Simple keyword-based routing
    internal_keywords = ["project", "confidential", "company", "revenue", 
                         "internal", "quarterly", "atlas", "financial"]
    
    if any(keyword in query for keyword in internal_keywords):
        route = "atlas"
        print(f"{BLUE}[ROUTER] Detected atlas query → routing to Atlas Agent{RESET}\n")
    else:
        route = "research"
        print(f"{YELLOW}[ROUTER] Detected research query → routing to Research Agent{RESET}\n")
    
    return {"route": route}

def atlas_agent_node(state: AgentState) -> AgentState:
    """Handle atlas queries using atlas vault."""
    print(f"{BLUE}[ATLAS AGENT] Retrieving from atlas vault...{RESET}")
    
    results = query_vault(ATLAS_VAULT, state["query"])
    context = "\n\n".join([r["text"] for r in results])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    messages = [
        SystemMessage(content="You are an internal company assistant with access to confidential information. Answer using ONLY the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}")
    ]
    
    response = llm.invoke(messages)
    
    print(f"{GREEN}[ATLAS AGENT] Answer generated{RESET}\n")
    
    return {
        "context": context,
        "answer": response.content
    }

def research_agent_node(state: AgentState) -> AgentState:
    """Handle research queries using research vault."""
    print(f"{YELLOW}[RESEARCH AGENT] Retrieving from research vault...{RESET}")
    
    results = query_vault(RESEARCH_VAULT, state["query"])
    context = "\n\n".join([r["text"] for r in results])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    messages = [
        SystemMessage(content="You are a research assistant. Answer using ONLY the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}")
    ]
    
    response = llm.invoke(messages)
    
    print(f"{GREEN}[RESEARCH AGENT] Answer generated{RESET}\n")
    
    return {
        "context": context,
        "answer": response.content
    }

# =====================================================================
# Graph
# =====================================================================

def create_graph():
    """Build LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("atlas_agent", atlas_agent_node)
    workflow.add_node("research_agent", research_agent_node)
    
    # Define edges
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        lambda x: x["route"],
        {
            "atlas": "atlas_agent",
            "research": "research_agent"
        }
    )
    
    workflow.add_edge("atlas_agent", END)
    workflow.add_edge("research_agent", END)
    
    return workflow.compile()

# =====================================================================
# Setup
# =====================================================================

def setup_vaults():
    """Initialize and index both vaults."""    
    print(f"{BLUE}[SETUP] Indexing atlas vault...{RESET}")
    index_vault(ATLAS_VAULT)
    
    print(f"{YELLOW}[SETUP] Indexing research vault...{RESET}")
    index_vault(RESEARCH_VAULT)
    
    print(f"{GREEN}[SETUP] Vaults ready!{RESET}\n")

# =====================================================================
# Main
# =====================================================================

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{YELLOW}ERROR: OPENAI_API_KEY not set{RESET}")
        return
    
    print("=" * 70)
    print("LangGraph Multi-Vault Demo")
    print("=" * 70)
    print()
    
    server = start_server()
    
    try:
        setup_vaults()
        
        graph = create_graph()
        
        queries = [
            "What are the key principles of quantum computing?",
            "What is Project Atlas and when is it launching?",
            "Explain how transformers work in neural networks",
            "What were our Q4 2024 financial results?"
        ]
        
        for query in queries:
            print("=" * 70)
            print(f"QUERY: {query}")
            print("=" * 70)
            print()
            
            result = graph.invoke({"query": query})
            
            print(f"{GREEN}ANSWER:{RESET}")
            print(result["answer"])
            print("\n")
    
    finally:
        server.terminate()
        print(f"{BLUE}[SERVER] Stopped{RESET}")

if __name__ == "__main__":
    main()