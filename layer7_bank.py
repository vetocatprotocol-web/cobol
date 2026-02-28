from protocol_bridge import TypedBuffer, ProtocolLanguage
import numpy as np
import struct

class Layer7Bank:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Pointers -> COMP-3 Packed Decimal (COBOL Bank Format)
        # Lossless: store length + binary data
        length = len(buffer.data)
        length_bytes = struct.pack('<I', length)
        comp3 = length_bytes + buffer.data.tobytes()
        return TypedBuffer.create(comp3, ProtocolLanguage.L7_COMP3, bytes)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Lossless: read length + reconstruct array
        length = struct.unpack('<I', buffer.data[:4])[0]
        pointers = np.frombuffer(buffer.data[4:4+length*4], dtype=np.uint32)
        return TypedBuffer.create(pointers, ProtocolLanguage.L6_PTR, np.ndarray)
