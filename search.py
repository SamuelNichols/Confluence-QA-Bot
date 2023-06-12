from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from tools.search import ConfluenceLinkSearchTool, ConfluencePageDescriberTool
from space_config import update_confluence_space

# setting confluence space
update_confluence_space("uProfile")

#initializing tools for agent
tools = [ConfluenceLinkSearchTool(), ConfluencePageDescriberTool()]
llm = OpenAI(temperature=0, model_name="text-davinci-003")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

with get_openai_callback() as cb:
    res = agent("give me links to articles that will help me mock an email in genesis")
    print(cb)