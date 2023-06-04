# This code sample uses the 'requests' library:
# http://docs.python-requests.org
import tempfile
import os
import timeit

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from util.cost import EmbeddingCostEstimator
from util.map_confluence import get_confluence_tree, get_path_to_root
from util.api import get_confluence_api, url
# args
space_arg = "uProfile"

# api variables
confluence_pat = os.environ.get("CONFLUENCE_PAT")

# intialized confluence api
api = get_confluence_api()

# getting all pages from uprofile
curr_len = 500
page_index = 0
all_pages = []
while curr_len == 500:
    pages = api.get_all_pages_from_space(space=space_arg, limit=500, start=page_index)
    curr_len = len(pages)
    page_index += curr_len
    all_pages += pages
print(len(all_pages), "pages found")

# getting tree structure of confluence from space
print("Getting confluence space tree structure")
confluence_root_tree = get_confluence_tree(page_title="User Profile Platform Home", space=space_arg)

# loading all pages into documents
cost = EmbeddingCostEstimator()
docs = []
print("Starting to load documents")
start_time = timeit.default_timer()
for i, page in enumerate(all_pages): 
    try:
        pdf = api.get_page_as_pdf(page_id=page['id'])
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True, dir='.') as temp_file:
            temp_file.write(pdf)
            temp_file.flush()  # Ensure the data is written to the file
            loader = PyPDFLoader(temp_file.name)
            pages = loader.load()
            for page in pages:
                cost.add_document(page)
                page.metadata['source'] = get_path_to_root(confluence_root_tree, page.metadata['title'])
                page.metadata['page_link'] = url + all_pages[i]['_links']['webui']
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
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
db = FAISS.from_documents(docs, embeddings)
elapsed = timeit.default_timer() - start_time
print("Finished embedding documents")
print("Time elapsed:", elapsed)
print("Saving to local faiss index")
db.save_local("confluence_faiss_index")
