from protocol_bridge import ProtocolLanguage
from typing import Dict, Any, Type
import numpy as np

class DictionaryManager:
    """
    Manages dictionaries for each layer with strict type support.
    Each dictionary specifies input/output type and ProtocolLanguage header.
    """
    
    def __init__(self, header: ProtocolLanguage, input_type: Type, output_type: Type):
        self.header = header
        self.input_type = input_type
        self.output_type = output_type
        self.dictionary: Dict[Any, Any] = {}
        self.reverse_dictionary: Dict[Any, Any] = {}
    
    def add(self, key: Any, value: Any) -> None:
        """Add key-value pair to dictionary"""
        self.dictionary[key] = value
        self.reverse_dictionary[value] = key
    
    def get(self, key: Any) -> Any:
        """Get value from dictionary"""
        return self.dictionary.get(key)
    
    def get_reverse(self, value: Any) -> Any:
        """Get key from reverse dictionary"""
        return self.reverse_dictionary.get(value)
    
    def contains(self, key: Any) -> bool:
        """Check if key exists"""
        return key in self.dictionary
    
    def size(self) -> int:
        """Get dictionary size"""
        return len(self.dictionary)
    
    def clear(self) -> None:
        """Clear dictionary"""
        self.dictionary.clear()
        self.reverse_dictionary.clear()
    
    def to_bytes(self) -> bytes:
        """Serialize dictionary to bytes"""
        import json
        data = {
            'header': self.header.name,
            'input_type': self.input_type.__name__,
            'output_type': self.output_type.__name__,
            'dictionary': {str(k): str(v) for k, v in self.dictionary.items()}
        }
        return json.dumps(data).encode()
    
    @staticmethod
    def from_bytes(data: bytes, header: ProtocolLanguage, input_type: Type, output_type: Type) -> 'DictionaryManager':
        """Deserialize dictionary from bytes"""
        import json
        dm = DictionaryManager(header, input_type, output_type)
        parsed = json.loads(data.decode())
        for k, v in parsed['dictionary'].items():
            dm.add(k, v)
        return dm

class DictionaryManagerL1(DictionaryManager):
    """L1 Semantic: Token_ID (uint8)"""
    def __init__(self):
        super().__init__(ProtocolLanguage.L1_SEM, str, np.ndarray)

class DictionaryManagerL2(DictionaryManager):
    """L2 Structural: Schema_Template_ID (uint16)"""
    def __init__(self):
        super().__init__(ProtocolLanguage.L2_STRUCT, np.ndarray, np.ndarray)

class DictionaryManagerL3(DictionaryManager):
    """L3 Delta: Signed_Delta_Integers (int16)"""
    def __init__(self):
        super().__init__(ProtocolLanguage.L3_DELTA, np.ndarray, np.ndarray)

class DictionaryManagerL4(DictionaryManager):
    """L4 Binary: Variable-Width Bitstream (bytes)"""
    def __init__(self):
        super().__init__(ProtocolLanguage.L4_BIN, np.ndarray, bytes)

class DictionaryManagerL5(DictionaryManager):
    """L5 Recursive: Nested_ID_Pointers (uint32)"""
    def __init__(self):
        super().__init__(ProtocolLanguage.L5_TRIE, bytes, np.ndarray)

class DictionaryManagerL6(DictionaryManager):
    """L6 Recursive: Nested_ID_Pointers (uint32)"""
    def __init__(self):
        super().__init__(ProtocolLanguage.L6_PTR, np.ndarray, np.ndarray)

class DictionaryManagerL7(DictionaryManager):
    """L7 Bank: COMP-3 Packed Decimal (bytes)"""
    def __init__(self):
        super().__init__(ProtocolLanguage.L7_COMP3, np.ndarray, bytes)

class DictionaryManagerL8(DictionaryManager):
    """L8 Final: COBOL Copybook Instruction (str, PIC X)"""
    def __init__(self):
        super().__init__(ProtocolLanguage.L8_COBOL, bytes, str)
