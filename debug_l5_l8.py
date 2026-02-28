"""Quick debug test untuk L5-L8 pipeline"""
from protocol_bridge import TypedBuffer, ProtocolLanguage
from layer5_recursive import Layer5Recursive
from layer6_recursive import Layer6Recursive
from layer7_bank import Layer7Bank
from layer8_final import Layer8Final

# Test L5-L8 saja
print("Testing L5-L8 chain...")

l5 = Layer5Recursive()
l6 = Layer6Recursive()
l7 = Layer7Bank()
l8 = Layer8Final()

# Input untuk L5 adalah bytes (output dari L4: Binary)
test_bytes = b"TEST BITSTREAM"

try:
    # L5: Bitstream -> Nested_ID_Pointers
    l5_buf = TypedBuffer.create(test_bytes, ProtocolLanguage.L4_BIN, bytes)
    print(f"L5 input: {l5_buf.data} (type: {type(l5_buf.data)})")
    
    l5_out = l5.encode(l5_buf)
    print(f"L5 output: {l5_out.data} (type: {type(l5_out.data)})")
    
    # L6: Nested_ID_Pointers -> Nested_ID_Pointers
    l6_out = l6.encode(l5_out)
    print(f"L6 output: {l6_out.data} (type: {type(l6_out.data)})")
    
    # L7: Nested_ID_Pointers -> COMP-3
    l7_out = l7.encode(l6_out)
    print(f"L7 output: {l7_out.data} (type: {type(l7_out.data)})")
    
    # L8: COMP-3 -> COBOL PIC X
    l8_out = l8.encode(l7_out)
    print(f"L8 output: {l8_out.data} (type: {type(l8_out.data)})")
    
    # Decompress L8->L7->L6->L5
    l7_back = l8.decode(l8_out)
    print(f"L8 back to L7: OK")
    
    l6_back = l7.decode(l7_back)
    print(f"L7 back to L6: OK")
    
    l5_back = l6.decode(l6_back)
    print(f"L6 back to L5: OK")
    
    l4_back = l5.decode(l5_back)
    print(f"L5 back to L4: OK")
    
    if l4_back.data == test_bytes:
        print("✓ LOSSLESS ROUNDTRIP SUCCESSFUL")
    else:
        print(f"✗ Data mismatch: {l4_back.data} != {test_bytes}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
