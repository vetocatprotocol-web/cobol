from dataclasses import dataclass
from enum import Enum
import hashlib
import numpy as np

class ProtocolLanguage(Enum):
    L1_SEM = 1
    L2_STRUCT = 2
    L3_DELTA = 3
    L4_BIN = 4
    L5_TRIE = 5
    L6_PTR = 6
    L7_COMP3 = 7
    L8_COBOL = 8

@dataclass
class TypedBuffer:
    data: any
    header: ProtocolLanguage
    type: type
    sha256: str

    @staticmethod
    def create(data, header, type_):
        sha = hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()
        return TypedBuffer(data, header, type_, sha)

class ProtocolBridge:
    def __init__(self, layers):
        self.layers = layers

    def compress(self, buffer: TypedBuffer) -> TypedBuffer:
        for layer in self.layers:
            buffer = layer.encode(buffer)
        return buffer

    def decompress(self, buffer: TypedBuffer) -> TypedBuffer:
        for layer in reversed(self.layers):
            buffer = layer.decode(buffer)
        return buffer
