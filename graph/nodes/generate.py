from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate_fn(state: GraphState) -> Dict[str, Any]:
    print("*******Generating*******")
    qstn = state["question"]
    docs = state["documents"]
    generation = generation_chain.invoke({
        "context": docs,
        "question": qstn
    })
    return {
        "documents": docs,
        "question": qstn,
        "generation": generation
    }