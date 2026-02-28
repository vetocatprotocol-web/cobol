from protocol_bridge import TypedBuffer, ProtocolLanguage
import numpy as np

class Layer3Delta:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Schema_IDs -> Signed_Delta_Integers
        deltas = np.diff(buffer.data, prepend=0).astype(np.int16)
        return TypedBuffer.create(deltas, ProtocolLanguage.L3_DELTA, np.ndarray)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        schema_ids = np.cumsum(buffer.data).astype(np.uint16)
        return TypedBuffer.create(schema_ids, ProtocolLanguage.L2_STRUCT, np.ndarray)
