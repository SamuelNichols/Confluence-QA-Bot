import tiktoken
from langchain.schema import Document

ADA_V2_TEXT_EMBED_COST = 0.0004
ADA_V2_TEXT_EMBED_TOKENS = 1000

class EmbeddingCostEstimator:
    def __init__(self):
        self.tokens = 0
        self.cost = 0

    def add_document(self, document: Document) -> None:
        tokens = self.num_tokens_from_string(document.page_content)
        self.tokens += tokens
        self.cost = self.embedding_cost_estimate(self.tokens, ADA_V2_TEXT_EMBED_COST, ADA_V2_TEXT_EMBED_TOKENS)
        
    def print_cost(self) -> None:
        print(f"Estimated cost: ${self.cost}")
        
    def print_tokens(self) -> None:
        print(f"Total tokens: {self.tokens}")

    def embedding_cost_estimate(self, num_tokens:int, model_cost: str, model_tokens: str) -> int:
        """Returns the estimated cost of an openai text embedding run
        Args:
            num_tokens: number of tokens in the text
            model_cost: cost of the model per 1000 tokens
            model_tokens: number of tokens the model can process
        Returns:
            estimated cost of the model
        """
        # calculate the cost then round to 4 decimal places
        cost = (num_tokens/model_tokens) * model_cost
        return float("{:.4f}".format(cost))

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(string))
        return num_tokens