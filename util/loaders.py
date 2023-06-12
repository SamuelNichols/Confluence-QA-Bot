import tempfile
from typing import List
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader

# load pdf binary as a list of langchain Document objects
def from_pdf(pdf: bytes) -> List[Document]:
    pages = []
    with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=True, dir="."
        ) as temp_file:
            temp_file.write(pdf)
            temp_file.flush()  # Ensure the data is written to the file
            loader = PyPDFLoader(temp_file.name)
            pages = loader.load() 
    return pages