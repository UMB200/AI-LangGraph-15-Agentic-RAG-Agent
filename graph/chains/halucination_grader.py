from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

class HalucinationGrader(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded on the facts 'yes' or 'no'.")

structured_llm_grader = llm.with_structured_output(
    HalucinationGrader, method='function_calling')

system_msg = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system_msg),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")])

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader