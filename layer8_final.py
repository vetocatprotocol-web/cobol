from protocol_bridge import TypedBuffer, ProtocolLanguage
import base64

class Layer8Final:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # COMP-3 -> COBOL Copybook Instruction (PIC X)
        # Lossless: use base64 encoding
        b64 = base64.b64encode(buffer.data).decode('ascii')
        pic_x = 'PIC X(' + str(len(buffer.data)) + ') VALUE IS \'' + b64 + '\''
        return TypedBuffer.create(pic_x, ProtocolLanguage.L8_COBOL, str)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Lossless: extract base64 and decode
        # Parse: PIC X(N) VALUE IS 'base64_here'
        try:
            start = buffer.data.find("'") + 1
            end = buffer.data.rfind("'")
            b64_str = buffer.data[start:end]
            comp3 = base64.b64decode(b64_str)
            return TypedBuffer.create(comp3, ProtocolLanguage.L7_COMP3, bytes)
        except Exception:
            # Fallback: return as bytes
            return TypedBuffer.create(buffer.data.encode(), ProtocolLanguage.L7_COMP3, bytes)
