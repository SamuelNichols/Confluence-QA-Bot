from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import Tool, initialize_agent
from langchain.schema import Document, HumanMessage
from langchain.callbacks import get_openai_callback
import tiktoken

class ConfluenceLinkSearchTool(BaseTool):
    name = "Confluence Search Tool"
    description = """given a question from related to uProfile, genesis, and Identity
    returns a list of links to confluence pages that might answer the question
    To use this tool you must provide the folloing parameter
    ['question']
    """

    # Getting vectorstore as retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    uprofile_db = FAISS.load_local("confluence_faiss_index", embeddings)
    urpofile_db_retriever = uprofile_db.as_retriever()

    def _run(self, query: str) -> str:
        """Use the tool."""
        results = self.uprofile_db.similarity_search(query, k=5)
        results_prompt = ""
        for i, result in enumerate(results):
            results_prompt += f"i: {i} _title: {result.metadata.get('title')} _link: {result.metadata.get('link')} _content: {result.page_content}\n"
        
        prompt_template = PromptTemplate(
            input_variables=["results_prompt", "query"],
            template="""Use the following information to answer the question, do not use any other information.\n
            {results_prompt}
            
            given the following question, create a list of links to confluence pages that might answer the question
            use the following format and make it human readable:
            1. Title (title of the confluence page that might answer the question, comes fomr _title)
            Link (link to the confluence page that might answer the question, comes from _link)
            Summary (summary of the confluence page that might answer the question, comes from _content)
            ...
            % QUESTION: {query}
            % RESPONSE:
            """
        )
        question = prompt_template.format(results_prompt=results_prompt, query=query)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        return llm(messages=[HumanMessage(content=question)]).content
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

def get_docs_from_doc(vectorstore: FAISS, document: Document) -> List[Document]:
    """every confluence page might be broken down into multiple documents
        returns all documents that share a page with a given document
    """
    similar = vectorstore.similarity_search(document.metadata.get('title'), k=10)
    pages = [doc for doc in similar if doc.metadata.get('title') == document.metadata.get('title')]
    return pages
    
# # Getting vectorstore as retriever
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
uprofile_db = FAISS.load_local("confluence_faiss_index", embeddings)

query = "where would I look to fix a pagerduty problem?"

results = uprofile_db.similarity_search(query, k=5)
results_prompt = ""
for i, result in enumerate(results):
    results_prompt += f"i: {i} _title: {result.metadata.get('title')} _link: {result.metadata.get('page_link')} _content: {result.page_content}\n"

prompt_template = PromptTemplate(
    input_variables=["results_prompt", "query"],
    template="""Use the following information to answer the question, do not use any other information.\n
    {results_prompt}
    
    given the following question, create a list of links to confluence pages that might answer the question
    use the following format and make it human readable:
    1. Title (title of the confluence page that might answer the question, comes fomr _title)
       Link (link to the confluence page that might answer the question, comes from _link)
       Summary (summary of the confluence page that might answer the question, comes from _content)
    ...
    % QUESTION: {query}
    % RESPONSE:
    """
)
question = prompt_template.format(results_prompt=results_prompt, query=query)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
with get_openai_callback() as cb:
    result = llm(messages=[HumanMessage(content=question)])
    print(result.content)
    print(cb.total_cost)
    
# TODO: add a tool that takes a title and returns the page as a set of steps or a detailed explaination+summary
# TODO: add an agent that puts together the steps of getting related links and applying the resulting pages to the initial question, will search for more context if needed
