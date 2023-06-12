import os
import timeit

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from atlassian import Confluence

from util.cost import EmbeddingCostEstimator
from util.api import get_confluence_api, url
from util.loaders import from_pdf

# args
space_arg = "uProfile"

# api variables
confluence_pat = os.environ.get("CONFLUENCE_PAT")

# intialized confluence api
api = get_confluence_api()

def get_all_pages_from_space_and_page(api: Confluence, page_id, all_pages=[]):
    """gets all pages from a given space and page"""
    if all_pages == []:
        new_page = api.get_page_by_id(page_id=page_id)
        all_pages.append(new_page)
    children = api.get_page_child_by_type(page_id=page_id, type='page')
    for page in children:
        all_pages.append(page)
        get_all_pages_from_space_and_page(api, page['id'], all_pages)
    return all_pages
    

# getting all pages from uprofile
curr_len = 500
page_index = 0
print("Getting all pages from uprofile")
home_page = api.get_page_by_title(space=space_arg, title="User Profile Platform Home")
all_pages = get_all_pages_from_space_and_page(api, home_page['id'])
print("Total number of pages retrieved: ", len(all_pages))

# loading all pages into documents
cost = EmbeddingCostEstimator()
docs = []
print("Starting to load documents")
start_time = timeit.default_timer()
for i, page in enumerate(all_pages):
    try:
        pdf = api.get_page_as_pdf(page_id=page["id"])
        pages = from_pdf(pdf)
        for page in pages:
            cost.add_document(page)
            page.metadata["source"] = url + all_pages[i].get('_links').get('webui')
            page.metadata.update(all_pages[i])
        docs += pages
    except Exception as e:
        print(e)
        continue

elapsed = timeit.default_timer() - start_time
print("Finished loading documents")
print("Time elapsed:", elapsed)

# printing estimated cost to embed all documents and total tokens
cost.print_cost()
cost.print_tokens()

# embedding all documents and saving to local faiss index
print("Starting to embed documents")
start_time = timeit.default_timer()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = FAISS.from_documents(docs, embeddings)
elapsed = timeit.default_timer() - start_time
print("Finished embedding documents")
print("Time elapsed:", elapsed)
print("Saving to local faiss index")
db.save_local("confluence_faiss_index")
