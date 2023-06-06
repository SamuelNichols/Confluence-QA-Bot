from treelib import Tree

from util.api import get_confluence_api, url

api = get_confluence_api()

def map_confluence_page(page, tree: Tree, parent: str):
    children = api.get_page_child_by_type(page_id=page['id'], type='page')
    for item in children:
        page_name = item['title']
        tree.create_node(page_name, page_name, parent=parent, data=url+item['_links']['webui'])
        map_confluence_page(item, tree, page_name)
        
def get_confluence_tree(page_title: str, space) -> Tree:
    # Get the root page  
    root_page = api.get_page_by_title(space=space, title=page_title)

    # Create a tree and add the root node
    tree = Tree()
    tree.create_node(root_page['title'], root_page['title'], data=url+root_page['_links']['webui'])

    # Map the Confluence page structure
    map_confluence_page(root_page, tree, root_page['title'])
    return tree

def get_path_to_root(tree, page_title: str) -> str:
    path_str = ""
    path = tree.rsearch(page_title)
    for node in reversed(list(path)):
        path_str += node + " > "
    return "directory path to page in confluence from space home page(root): " + path_str[:-3]

print(get_path_to_root(get_confluence_tree("uProfile - Genesis Global", "uProfile"), "Automatic Schema Migration in CICD"))
    