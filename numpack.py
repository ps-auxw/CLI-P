import struct

def i2b(i, postfix=None):
    packed = struct.pack('<Q', i)
    if postfix is not None:
        packed += postfix
    return packed

def b2i(b):
    return struct.unpack('<Q', b[0:8])[0]

def f2b(i):
    return struct.pack('<f', i)

def b2f(b):
    return struct.unpack('<f', b[0:4])[0]

def s2b(i):
    return struct.pack('<H', i)

def b2s(b):
    return struct.unpack('<H', b[0:2])[0]

