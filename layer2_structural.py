from protocol_bridge import TypedBuffer, ProtocolLanguage
import numpy as np

class Layer2Structural:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Token_IDs -> Schema_Template_ID (Hex)
        # Dummy: group tokens into schema IDs (hex)
        schema_ids = np.array([t ^ 0xA5 for t in buffer.data], dtype=np.uint16)
        return TypedBuffer.create(schema_ids, ProtocolLanguage.L2_STRUCT, np.ndarray)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        tokens = np.array([s ^ 0xA5 for s in buffer.data], dtype=np.uint8)
        return TypedBuffer.create(tokens, ProtocolLanguage.L1_SEM, np.ndarray)
