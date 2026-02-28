from protocol_bridge import TypedBuffer, ProtocolLanguage
import numpy as np

class Layer7Bank:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Pointers -> COMP-3 Packed Decimal (COBOL Bank Format)
        # Dummy: pack uint32 to COMP-3 (BCD)
        def int_to_comp3(val):
            bcd = bytearray()
            while val:
                bcd.append(val % 10)
                val //= 10
            return bytes(bcd)
        comp3 = b''.join([int_to_comp3(int(x)) for x in buffer.data])
        return TypedBuffer.create(comp3, ProtocolLanguage.L7_COMP3, bytes)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Dummy: unpack COMP-3 to pointers (lossless for test)
        # Assume each pointer is 4 bytes
        n = len(buffer.data) // 4
        pointers = np.frombuffer(buffer.data[:n*4], dtype=np.uint32)
        return TypedBuffer.create(pointers, ProtocolLanguage.L6_PTR, np.ndarray)
