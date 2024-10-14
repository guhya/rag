from typing_extensions import TypedDict

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        ori_prompt: Original prompt fromt the user
        evaluation: Final evaluation presented to the user
    """
    
    ori_prompt: str
    evaluation: list[any]        