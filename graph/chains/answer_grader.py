from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

class AnswerGrader(BaseModel):
    """Binary score for answer correctness."""
    binary_score: bool = Field(
        description="Answer is correct, 'yes' or 'no'.")

llm = ChatOpenAI(temperature=0.0)

structured_llm_grader = llm.with_structured_output(AnswerGrader, method='function_calling')

system_msg = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_msg),
    ("human", "User question: {question} \n\n LLM generation: {generation}")])

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader