from protocol_bridge import TypedBuffer, ProtocolLanguage
import numpy as np

class Layer4Binary:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Deltas -> Variable-Width Bitstream
        # Dummy: pack int16 to bytes
        bitstream = buffer.data.tobytes()
        return TypedBuffer.create(bitstream, ProtocolLanguage.L4_BIN, bytes)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        deltas = np.frombuffer(buffer.data, dtype=np.int16)
        return TypedBuffer.create(deltas, ProtocolLanguage.L3_DELTA, np.ndarray)
