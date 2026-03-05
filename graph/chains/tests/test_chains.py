from dotenv import load_dotenv
from pprint import pprint
load_dotenv()

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.halucination_grader import hallucination_grader, HalucinationGrader
from graph.chains.router import question_router_chain, QueryRouter

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    result: GradeDocuments = retrieval_grader.invoke({
        "question": question, 
        "document": doc_txt
    })
    assert result.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    result: GradeDocuments = retrieval_grader.invoke({
        "question": "how to bake a chocolate cake", 
        "document": doc_txt
    })
    assert result.binary_score == "no"

def test_generation_chain() -> None:
    qstn = "agent memory"
    docs = retriever.invoke(qstn)
    generation = generation_chain.invoke({
        "context": "\n\n".join([d.page_content[:500] for d in docs]), 
        "question": qstn}) 
    pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({
        "context": docs, 
        "question": question}) 
    result: HalucinationGrader = hallucination_grader.invoke({
        "documents": docs, 
        "generation": generation
    })
    assert result.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    result: HalucinationGrader = hallucination_grader.invoke({
        "documents": docs, 
        "generation": "In order to make pizza we need to first start with the dough"
    })
    assert not result.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"
    result: QueryRouter = question_router_chain.invoke({"question": question})
    assert result.datasource == "vectorstore"

def test_router_to_web_search() -> None:
    question = "What is the weather in New York today?"
    result: QueryRouter = question_router_chain.invoke({"question": question})
    assert result.datasource == "web_search"
