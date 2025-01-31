from typing import List, TypedDict
from xml.dom.minidom import Document


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
