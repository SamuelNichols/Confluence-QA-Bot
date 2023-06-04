# This code sample uses the 'requests' library:
# http://docs.python-requests.org
import os

from atlassian import Confluence

# api variables
confluence_pat = os.environ.get("CONFLUENCE_PAT")

url = "https://confluence.unity3d.com"

def get_confluence_api() -> Confluence:
    return Confluence(url=url, token=confluence_pat)