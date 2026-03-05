from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class QueryRouter(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.")

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
structured_router = llm.with_structured_output(
    QueryRouter, method='function_calling')
system_msg =  """You are an expert at routing a user question to a vectorstore or web search.
                The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
                Use the vectorstore for questions on these topics. For all else, use web-search."""
router_prompt = ChatPromptTemplate.from_messages([
    ("system", system_msg),
    ("human", "User question: {question}")])
question_router_chain = router_prompt | structured_router