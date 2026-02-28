from protocol_bridge import TypedBuffer, ProtocolLanguage
import numpy as np

class Layer5Recursive:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Bitstream -> Nested_ID_Pointers (Recursive Trie)
        # Dummy: treat each byte as pointer id
        pointers = np.array(list(buffer.data), dtype=np.uint32)
        return TypedBuffer.create(pointers, ProtocolLanguage.L5_TRIE, np.ndarray)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        bitstream = buffer.data.astype(np.uint8).tobytes()
        return TypedBuffer.create(bitstream, ProtocolLanguage.L4_BIN, bytes)
