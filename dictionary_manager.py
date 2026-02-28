from protocol_bridge import ProtocolLanguage
from typing import Dict, Any, Type
import numpy as np

class DictionaryManager:
    """
    Advanced Dictionary Manager for all layers (L1-L8).
    Fitur: adaptive learning, backup/versioning, hash chaining antar layer, global registry, DictionaryChain, statistik, dan serialization multi-node.
    """
    def __init__(self, header: ProtocolLanguage, input_type: Type, output_type: Type):
        self.header = header
        self.input_type = input_type
        self.output_type = output_type
        self.dictionary: Dict[Any, Any] = {}
        self.reverse_dictionary: Dict[Any, Any] = {}
        self.backup_dictionaries: Dict[str, list] = {}
        self.dictionary_hashes: Dict[str, bytes] = {}
        self.global_registry: Dict[str, Any] = {}
        self.dictionary_chain: list = []
        self.usage_stats: Dict[str, int] = {}
        self.version: int = 1
        self._initialize_base_dictionary()

    def _initialize_base_dictionary(self):
        """Inisialisasi dictionary dasar dan chain."""
        self.dictionary_chain.append(self.dictionary)

    def add(self, key: Any, value: Any) -> None:
        self.dictionary[key] = value
        self.reverse_dictionary[value] = key
        self.usage_stats[str(key)] = self.usage_stats.get(str(key), 0) + 1

    def get(self, key: Any) -> Any:
        self.usage_stats[str(key)] = self.usage_stats.get(str(key), 0) + 1
        return self.dictionary.get(key)

    def get_reverse(self, value: Any) -> Any:
        return self.reverse_dictionary.get(value)

    def contains(self, key: Any) -> bool:
        return key in self.dictionary

    def size(self) -> int:
        return len(self.dictionary)

    def clear(self) -> None:
        self.dictionary.clear()
        self.reverse_dictionary.clear()
        self.usage_stats.clear()

    def backup(self) -> None:
        """Backup dictionary versi saat ini."""
        import copy
        key = f"v{self.version}"
        self.backup_dictionaries[key] = copy.deepcopy(self.dictionary)
        self.version += 1

    def restore(self, version: int) -> None:
        key = f"v{version}"
        if key in self.backup_dictionaries:
            self.dictionary = self.backup_dictionaries[key]
            self.reverse_dictionary = {v: k for k, v in self.dictionary.items()}

    def update_hash(self, layer_name: str) -> None:
        """Update hash untuk chaining antar layer."""
        import hashlib
        dict_bytes = str(self.dictionary).encode()
        self.dictionary_hashes[layer_name] = hashlib.sha256(dict_bytes).digest()

    def register_global(self, name: str, obj: Any) -> None:
        self.global_registry[name] = obj

    def chain_dictionary(self, next_dict: Dict[Any, Any]) -> None:
        self.dictionary_chain.append(next_dict)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "size": self.size(),
            "usage_stats": self.usage_stats,
            "version": self.version,
            "backup_versions": list(self.backup_dictionaries.keys()),
            "hashes": {k: v.hex() for k, v in self.dictionary_hashes.items()},
            "chain_length": len(self.dictionary_chain),
            "global_registry_keys": list(self.global_registry.keys()),
        }

    def to_bytes(self) -> bytes:
        import json
        data = {
            'header': self.header.name,
            'input_type': self.input_type.__name__,
            'output_type': self.output_type.__name__,
            'dictionary': {str(k): str(v) for k, v in self.dictionary.items()},
            'version': self.version,
            'usage_stats': self.usage_stats,
            'backup_versions': list(self.backup_dictionaries.keys()),
            'hashes': {k: v.hex() for k, v in self.dictionary_hashes.items()},
            'chain_length': len(self.dictionary_chain),
            'global_registry_keys': list(self.global_registry.keys()),
        }
        return json.dumps(data).encode()

    @staticmethod
    def from_bytes(data: bytes, header: ProtocolLanguage, input_type: Type, output_type: Type) -> 'DictionaryManager':
        import json
        dm = DictionaryManager(header, input_type, output_type)
        parsed = json.loads(data.decode())
        for k, v in parsed['dictionary'].items():
            dm.add(k, v)
        dm.version = parsed.get('version', 1)
        dm.usage_stats = parsed.get('usage_stats', {})
        # backup_versions, hashes, chain_length, global_registry_keys dapat diisi sesuai kebutuhan
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
