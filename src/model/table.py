from dataclasses import dataclass, field

from src.model.element import Element
from typing import List

from src.utils.utils import Utils


@dataclass
class Table(Element):
    rows: List[List[str]] = field(default_factory=list)

    def __post_init__(self):
        self.element_type = Utils.TYPE_TABLE