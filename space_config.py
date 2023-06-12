confluence_space = None

def get_confluence_space():
    return confluence_space

def update_confluence_space(space):
    global confluence_space
    confluence_space = space