from dotenv import load_dotenv
from pprint import pprint
load_dotenv()

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain

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
    qstn = "agent_memory"
    docs = retriever.invoke(qstn)
    generation = generation_chain.invoke({
        "context": "\n\n".join([d.page_content[:500] for d in docs]), 
        "question": qstn}) 
    pprint(generation)