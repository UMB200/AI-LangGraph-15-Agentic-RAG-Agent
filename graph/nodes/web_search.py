from typing import Any, Dict
from langchain_core.documents import Document 
from langchain_tavily import TavilySearch 
from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()

tavily_search_tool = TavilySearch(max_results=3)

def web_search(state:GraphState) -> Dict[str, Any]:
    print("*******Web Search*******")
    qstn = state["question"]
    doc = state["documents"]

    tavily_res = tavily_search_tool.invoke({"query": qstn})
    if isinstance(tavily_res, dict):
        results_list = tavily_res.get("results", [])
    else:
        results_list = tavily_res
    
    joined_tavily_res = "\n".join(
        [tavily_r["content"] for tavily_r in results_list] 
    )
    search_result = Document(page_content=joined_tavily_res)
    if doc is not None:
        doc.append(search_result)
    else:
        doc = [search_result]
    return {"documents": doc, "question": qstn}

if __name__ == "__main__":
    output = web_search(state={"question": "agent_memory", "documents": None})
    print(output)