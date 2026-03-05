from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.const import RETRIEVE, GRADE_DOCS, GENERATE, WEBSEARCH
from graph.nodes import generate_fn, grade_docs, retrieve, web_search
from graph.chains.answer_grader import answer_grader
from graph.state import GraphState
from graph.chains.halucination_grader import hallucination_grader
from graph.chains.router import question_router_chain, QueryRouter
load_dotenv()

def decide_to_generate(state):
    print("********Assess graded docs*********")
    if state["web_search"]:
        print("****DECISION: not all docs are relevant to questiom, include web search***")
        return WEBSEARCH
    else:
        print("*****Decision: to generate******")
        return GENERATE
    
def grade_generation_in_docs_and_question(state: GraphState)-> str:
    print("*****Check hallucionations******")
    qstn = state["question"]
    docs = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({
        "documents": docs,
        "generation": generation
    })
    if hallucination_grade := score.binary_score:
        print("****DECISION: generation is based on the docs*****")
        print("*****Grade generation vs Question")
        score = answer_grader.invoke({
            "question": qstn,
            "generation": generation        })
        if answer_grade := score.binary_score:
            print("****DECISION: generation is correct answer to question*****")
            return "useful"
        else:
            print("****DECISION: generation is NOT correct answer to question*****")
            return "not useful"
    else:
        print("****DECISION: generation is NOT based on the docs*****")
        return "not supported"

def route_question(state: GraphState)-> str:
    print("*********Routing question**********")
    qstn = state["question"]
    source: QueryRouter = question_router_chain.invoke({"question": qstn})
    if source.datasource == "WEBSEARCH":
        print("****ROUTER DECISION: route question to web search*****")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("****ROUTER DECISION: route question to RAG*****")
        return RETRIEVE


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCS, grade_docs)
workflow.add_node(GENERATE, generate_fn)
workflow.add_node(WEBSEARCH, web_search)
workflow.set_conditional_entry_point(route_question, {
    WEBSEARCH: WEBSEARCH,
    RETRIEVE: RETRIEVE
})
workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCS)
workflow.add_conditional_edges(
    GRADE_DOCS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_in_docs_and_question,
    {"not supported": GENERATE, "useful": END, "not useful": WEBSEARCH}
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
