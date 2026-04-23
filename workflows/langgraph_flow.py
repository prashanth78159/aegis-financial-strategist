from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# -------------------------------
# State Definition
# -------------------------------

class AgentState(TypedDict):
    query: str
    rag_answer: Optional[str]
    web_answer: Optional[str]
    quant_answer: Optional[str]
    final_answer: Optional[str]

# -------------------------------
# Agent Nodes
# -------------------------------

def router(state: AgentState) -> AgentState:
    return {**state}

# --- RAG Analyst ---
def rag_analyst(state: AgentState) -> AgentState:
    from rag.vector_store import get_vector_store

    db = get_vector_store()
    docs = db.similarity_search(state['query'], k=5)

    if not docs:
        return {**state, 'rag_answer': '[RAG] No relevant SEC chunks retrieved.'}

    parts = []
    for d in docs:
        parts.append('[Page {}] {}'.format(d.metadata.get('page', 'N/A'), d.page_content))

    context = '\\n\\n'.join(parts)

    return {**state, 'rag_answer': context}

# --- Web Researcher ---
def web_researcher(state: AgentState) -> AgentState:
    from tavily import TavilyClient
    from config.settings import TAVILY_API_KEY

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=state['query'], max_results=3, search_depth='advanced')

    if not response or not response.get('results'):
        return {**state, 'web_answer': '[WEB] No relevant news found.'}

    parts = []
    for r in response['results']:
        parts.append('- {} ({})\\n  {}'.format(
            r.get('title', 'No title'),
            r.get('url', ''),
            r.get('content', '')
        ))

    return {**state, 'web_answer': '\\n\\n'.join(parts)}

# --- Python Quant (Revenue + Risk Signals) ---
def python_quant(state: AgentState) -> AgentState:
    import re

    rag_text = state.get('rag_answer', '')

    if not rag_text or rag_text.startswith('[RAG]'):
        return {**state, 'quant_answer': '[QUANT] No RAG financial context available.'}

    revenue_lines = [
        line for line in rag_text.split('\\n')
        if 'revenue' in line.lower() and '$' in line
    ]

    revenue_values = []
    if revenue_lines:
        numbers = re.findall(r'\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?', ' '.join(revenue_lines))
        revenue_values = [float(n.replace('$','').replace(',','')) for n in numbers]

    yoy = None
    if len(revenue_values) >= 2 and revenue_values[1] != 0:
        yoy = ((revenue_values[0] - revenue_values[1]) / revenue_values[1]) * 100

    risk_terms = [
        'liabilities','debt','borrowings','supply chain','regulatory',
        'litigation','privacy','data security','inflation','interest rates'
    ]

    detected_risks = [k for k in risk_terms if k in rag_text.lower()]

    parts = []
    if revenue_values:
        parts.append('Extracted Revenue Figures: {}'.format(revenue_values[:3]))
        if yoy is not None:
            parts.append('YoY Revenue Growth: {:.2f}%'.format(yoy))
    else:
        parts.append('Revenue figures not found in retrieved SEC context.')

    if detected_risks:
        parts.append('Risk Signals Detected: {}'.format(', '.join(sorted(set(detected_risks)))))

    return {**state, 'quant_answer': '[QUANT] ' + ' | '.join(parts)}

# --- Auditor ---
def auditor(state: AgentState) -> AgentState:
    final = (
        'RAG Findings:\\n' + str(state.get('rag_answer')) + '\\n\\n' +
        'Web Findings:\\n' + str(state.get('web_answer')) + '\\n\\n' +
        'Quant Analysis:\\n' + str(state.get('quant_answer'))
    )

    return {**state, 'final_answer': final}

# -------------------------------
# Graph Builder
# -------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node('router', router)
    graph.add_node('rag_analyst', rag_analyst)
    graph.add_node('web_researcher', web_researcher)
    graph.add_node('python_quant', python_quant)
    graph.add_node('auditor', auditor)

    graph.set_entry_point('router')
    graph.add_edge('router', 'rag_analyst')
    graph.add_edge('rag_analyst', 'web_researcher')
    graph.add_edge('web_researcher', 'python_quant')
    graph.add_edge('python_quant', 'auditor')
    graph.add_edge('auditor', END)

    return graph.compile()