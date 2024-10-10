import struct
import math
import numpy

# Use numpy.float32() to cast python float to 32 bit float as shown below
def float_HEX_repr(x):
    return hex(struct.unpack('<I', struct.pack('<f', x))[0])

# Used to retrieve hexadecimal representation of python reals
def python_HEX_repr(x):
    return hex(struct.unpack('<Q', struct.pack('<d', x))[0])

x = input("Input a real number: ")
x = float(x)
z = numpy.float32(x)

print("Hexadecimal representation (64-bits): "+str(python_HEX_repr(x)))
print("Hexadecimal representation (32-bits): "+str(float_HEX_repr(z)))
