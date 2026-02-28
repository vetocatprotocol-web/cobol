from protocol_bridge import TypedBuffer, ProtocolLanguage
import numpy as np

class Layer1Semantic:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Raw text -> Token_ID (np.uint8)
        tokens = np.array([ord(c) % 256 for c in buffer.data], dtype=np.uint8)
        return TypedBuffer.create(tokens, ProtocolLanguage.L1_SEM, np.ndarray)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        text = ''.join([chr(t) for t in buffer.data])
        return TypedBuffer.create(text, ProtocolLanguage.L1_SEM, str)
