from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    
@dataclass
class DocumentElement:
    id: str
    element_type: ElementType
    content: str
    summary: Optional[str] = None
    metadata: Optional[dict] = None