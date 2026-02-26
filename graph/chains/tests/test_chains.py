from dotenv import load_dotenv
load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

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