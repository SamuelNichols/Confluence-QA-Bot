from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

from tools.search import ConfluenceLinkSearchTool

# initializing tools for agent
tools = [ConfluenceLinkSearchTool()]
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# print("result: ", agent.run("need information about onboarding onto genesis"))

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
uprofile_db = FAISS.load_local("confluence_faiss_index", embeddings)
results = uprofile_db.similarity_search("uProfile Eng Onboarding - Genesis Onboarding", 5)
print("results: ", results)
for doc in results:
    print("doc: ", doc.metadata)
    if doc.metadata.get("title") == "uProfile Eng Onboarding - Genesis Onboarding":
        print("title: ", doc.metadata.get("title"))
    
# TODO: add a tool that takes a title and returns the page as a set of steps or a detailed explaination+summary

# TODO: add an agent that puts together the steps of getting related links and applying the resulting pages to the initial question, will search for more context if needed
