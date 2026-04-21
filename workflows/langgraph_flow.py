from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import os, requests, json

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_KEY = os.environ.get('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')

def call_llm(prompt: str) -> str:
    headers = {
        'Authorization': f'Bearer {OPENROUTER_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'meta-llama/llama-3.1-8b-instruct:free',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 500
    }
    res = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()['choices'][0]['message']['content']

class AgentState(TypedDict):
    query: str
    route: Optional[str]
    rag_answer: Optional[str]
    web_answer: Optional[str]
    quant_answer: Optional[str]
    final_answer: Optional[str]

def router(state: AgentState) -> AgentState:
    prompt = f'Classify into rag, rag_web, rag_quant, full: {state['query']}'
    out = call_llm(prompt).strip().lower()
    return {**state, 'route': out}

def rag_analyst(state: AgentState) -> AgentState:
    from rag.vector_store import get_vector_store
    db = get_vector_store()
    docs = db.similarity_search(state['query'], k=5)
    txt = '\n\n'.join(d.page_content for d in docs)
    return {**state, 'rag_answer': txt}

def web_researcher(state: AgentState) -> AgentState:
    from tavily import TavilyClient
    from config.settings import TAVILY_API_KEY
    tv = TavilyClient(api_key=TAVILY_API_KEY)
    out = tv.search(state['query'], max_results=3)
    txt = '\n\n'.join(r['content'] for r in out['results'])
    return {**state, 'web_answer': txt}

def python_quant(state: AgentState) -> AgentState:
    return {**state, 'quant_answer': 'Quant agent placeholder'}

def auditor(state: AgentState) -> AgentState:
    prompt = 'Analyze: ' + json.dumps(state)
    final = call_llm(prompt)
    return {**state, 'final_answer': final}

def build_graph():
    g = StateGraph(AgentState)
    g.add_node('router', router)
    g.add_node('rag', rag_analyst)
    g.add_node('web', web_researcher)
    g.add_node('quant', python_quant)
    g.add_node('auditor', auditor)
    g.set_entry_point('router')

    g.add_conditional_edges(
        'router', lambda s: s['route'],
        {'rag':'rag','rag_web':'rag','rag_quant':'rag','full':'rag'}
    )

    g.add_edge('rag', 'web')
    g.add_edge('web', 'quant')
    g.add_edge('quant', 'auditor')
    g.add_edge('auditor', END)

    return g.compile()