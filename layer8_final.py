from protocol_bridge import TypedBuffer, ProtocolLanguage

class Layer8Final:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # COMP-3 -> COBOL Copybook Instruction (PIC X)
        # Dummy: convert bytes to PIC X string
        pic_x = 'PIC X(' + str(len(buffer.data)) + ') ' + buffer.data.hex().upper()
        return TypedBuffer.create(pic_x, ProtocolLanguage.L8_COBOL, str)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Dummy: extract bytes from PIC X string
        hexstr = buffer.data.split(') ')[-1]
        comp3 = bytes.fromhex(hexstr)
        return TypedBuffer.create(comp3, ProtocolLanguage.L7_COMP3, bytes)
