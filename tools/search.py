from typing import List
import tiktoken
from atlassian import Confluence
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import BaseTool
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain

from util.api import get_confluence_api
from util.loaders import from_pdf
from space_config import get_confluence_space
    
class ConfluenceLinkSearchTool(BaseTool):
    name = "Confluence Search Tool"
    description = """useful for when you want to get a list of links to confluence pages that might answer a question
    """

    # Getting vectorstore
    uprofile_db: FAISS = FAISS.load_local("confluence_faiss_index", OpenAIEmbeddings(model="text-embedding-ada-002"))
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        results = self.uprofile_db.similarity_search(query, k=5)
        results_prompt = ""
        for i, result in enumerate(results):
            results_prompt += f"i: {i} _title: {result.metadata.get('title')} _source: {result.metadata.get('source')} _content: {result.page_content}\n"

        prompt_template = PromptTemplate(
            input_variables=["results_prompt", "query"],
            template="""Use the following information to answer the question
            do not use any other information.
            be clear and consise.
            {results_prompt}
            
            create a list of links to confluence pages that might answer the question
            use the following format
            [
                {{
                    "Title": "title = _title",
                    "Summary": "summary of _content, retains all context",
                    "Link": "link = _source"
                }},
                ...
            ] 

            % QUESTION: {query}
            % RESPONSE:
            """
        )
        question = prompt_template.format(results_prompt=results_prompt, query=query)
        llm = OpenAI(temperature=0, model_name="text-davinci-003", verbose=True)
        return llm(question)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

    
class ConfluencePageDescriberTool(BaseTool):
    name  = "Confluence Page Describer"
    description = """useful for when you want to get more information about a confluence page given its title and space
    """

    # getting confluence api
    api: Confluence = get_confluence_api()
    
    # Getting vectorstore
    uprofile_db: FAISS = FAISS.load_local("confluence_faiss_index", OpenAIEmbeddings(model="text-embedding-ada-002"))

    def _run(self, title: str) -> List[str]:
        """Use the tool."""
        curr_page = self.api.get_page_by_title(get_confluence_space(), title)
        page_pdf = self.api.get_page_as_pdf(curr_page['id'])
        pages = from_pdf(page_pdf)
        # llm = OpenAI(temperature=0, model_name="text-davinci-003", verbose=True)
        # chain = load_qa_chain(llm = llm, chain_type="refine")
        # question = """
        # keep as much instructional information as possible while keeping the page under 3000 tokens
        # """
        # result = (chain({"input_documents": pages, "question": question}, return_only_outputs=True)['output_text'])
        # print(result)
        encoding = tiktoken.get_encoding("cl100k_base")
        concatenated_string = ''.join([doc.page_content for doc in pages])
        num_tokens = len(encoding.encode(concatenated_string))
        return f"tokens in page: {num_tokens}"
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    