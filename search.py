from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import Tool, initialize_agent
from pydantic import BaseModel


# class ConfluenceSearchTool(BaseTool):
#     name = "Confluence Search Tool"
#     description = """searches confluence for articles related to uProfile, genesis, and Identity
#     To use this tool you must provide the folloing parameter
#     ['query']
#     """

#     # Getting vectorstore as retriever
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#     uprofile_db = FAISS.load_local("confluence_faiss_index", embeddings)
#     urpofile_db_retriever = uprofile_db.as_retriever()
#     # Creating prompt for qa chain
#     search_template = PromptTemplate(
#         input_variables=["question"],
#         template="""
#                         % QUERY:
#                         Using the information given, do your best to answer the question
#                         Format the answer in an easy to read way
#                         Give as much detail as possible to give a complete answer
#                         after you finish your answer, present links to all relavant articles

#                         % REQUEST: {question}
                        
#                         % RESPONSE:
#                     """,
#     )
#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
#     # retrieval qa chain
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=urpofile_db_retriever,
#         verbose=True,
#     )

#     def _run(self, query: str) -> str:
#         """Use the tool."""
#         question = self.search_template.format(question=query)
#         return self.qa.run(question)
    
#     async def _arun(self, query: str) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError("custom_search does not support async")


# # Initializing qa chain
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# # # Initializing tool
# tools = [ConfluenceSearchTool()]

# # Initializing agent
# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# agent = initialize_agent(
#     agent='chat-conversational-react-description',
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method='generate',
#     memory=memory
# )

# user_input = None
# while user_input != "exit":
#     if user_input is not None:
#         agent.run(user_input)
#     print("Enter a question to search confluence for: ")
#     user_input = input()

# Getting vectorstore as retriever
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
uprofile_db = FAISS.load_local("confluence_faiss_index", embeddings)
urpofile_db_retriever = uprofile_db.as_retriever()
# Creating prompt for qa chain
search_template = PromptTemplate(
    input_variables=["question"],
    template="""
                    % QUERY:
                    Using the information given, do your best to answer the question
                    Format the answer in an easy to read way
                    Give as much detail as possible to give a complete answer
                    after you finish your answer, present links to all relavant articles

                    % REQUEST: {question}
                    
                    % RESPONSE:
                """,
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# retrieval qa chain
qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=urpofile_db_retriever,
    verbose=True,
)
print(qa(search_template.format(question="What is uProfile?")))
    
