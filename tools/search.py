from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, Document

from util.api import get_confluence_api

def get_docs_from_doc(vectorstore: FAISS, document: Document) -> List[Document]:
    """every confluence page might be broken down into multiple documents
        returns all documents that share a page with a given document
    """
    similar = vectorstore.similarity_search(document.metadata.get('title'), k=10)
    pages = [doc for doc in similar if doc.metadata.get('title') == document.metadata.get('title')]
    return pages

class ConfluenceLinkSearchTool(BaseTool):
    name = "Confluence Search Tool"
    description = """given a question
    returns a list of links to confluence pages that might answer the question
    """
    # return_direct = True

    # Getting vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    uprofile_db = FAISS.load_local("confluence_faiss_index", embeddings)

    def _run(self, query: str) -> str:
        """Use the tool."""
        results = self.uprofile_db.similarity_search(query, k=5)
        results_prompt = ""
        for i, result in enumerate(results):
            results_prompt += f"i: {i} _title: {result.metadata.get('title')} _source: {result.metadata.get('source')} _content: {result.page_content}\n"

        prompt_template = PromptTemplate(
            input_variables=["results_prompt", "query"],
            template="""Use the following information to answer the question, do not use any other information.\n
            {results_prompt}
            
            given the following question, create a list of links to confluence pages that might answer the question
            use the following format and make it human readable:
            1. Title (title of the confluence page that might answer the question, comes fomr _title)
            Summary (summary of the confluence page that might answer the question, comes from _content)
            Link (link to the confluence page that might answer the question, comes from _source)
            ...
            % QUESTION: {query}
            % RESPONSE:
            """
        )
        question = prompt_template.format(results_prompt=results_prompt, query=query)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
        return llm(messages=[HumanMessage(content=question)]).content
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
class ConfluenceLinkDescriber(BaseTool):
    name = "Confluence Search Tool"
    description = """given a confluence page title
    gathers all information about that page and returns a summary that retains all of the original context
    """
    # return_direct = False

    # getting confluence api
    api = get_confluence_api()

    def _run(self, page_title: str) -> str:
        """Use the tool."""
        self.api.get()

        prompt_template = PromptTemplate(
            input_variables=["results_prompt", "query"],
            template="""Use the following information to answer the question, do not use any other information.\n
            {results_prompt}
            
            given the following question, create a list of links to confluence pages that might answer the question
            use the following format and make it human readable:
            1. Title (title of the confluence page that might answer the question, comes fomr _title)
            Summary (summary of the confluence page that might answer the question, comes from _content)
            Link (link to the confluence page that might answer the question, comes from _source)
            ...
            % QUESTION: {query}
            % RESPONSE:
            """
        )
        question = prompt_template.format(results_prompt=results_prompt, query=query)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
        return llm(messages=[HumanMessage(content=question)]).content
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    