import lmdb
import numpy as np
import hashlib
import struct
from numpack import s2b, b2s

# LMDB environment
env = None

# Filename hash to index
fn_db = None

# Filenames that were skipped
skip_db = None

# Fixed mapping from uint64 to filename and to vector (+'n' for filename, +'v' for vector, +'f' for face count, +'f'+fidx for face data)
fix_idx_db = None

# faiss index to fixed index
idx_db = None

# faiss face index to faiss index and face index in image
idx_face_db = None

# uint64
int_type = '<Q'

def open_db(map_size=1024*1024*1024*20, pack_type='<Q'):
    global env, fn_db, skip_db, fix_idx_db, idx_db, idx_face_db, int_type
    env = lmdb.open('vectors.lmdb', map_size=map_size, max_dbs=5)

    fn_db = env.open_db(b'fn_db')
    skip_db = env.open_db(b'skip_db')
    fix_idx_db = env.open_db(b'fix_idx_db')
    idx_db = env.open_db(b'idx_db')
    idx_face_db = env.open_db(b'idx_face_db')
    int_type = pack_type

    with env.begin(db=fix_idx_db, write=True) as txn:
        res = txn.get(b'int_type')
        if res is None:
            txn.put(b'int_type', pack_type.encode())
        else:
            int_type = res.decode()
        res = txn.get(b'next')
        if res is None:
            txn.put(b'next', i2b(0))

def key_len():
    l = 8
    if int_type == '<L':
        l = 4
    return l

def i2b(i, postfix=None):
    packed = struct.pack(int_type, i)
    if postfix is not None:
        packed += postfix
    return packed

def b2i(b):
    return struct.unpack(int_type, b[0:key_len()])[0]

def get_next_idx():
    with env.begin(db=fix_idx_db) as txn:
        return txn.get(b'next')

def get_s(s, db):
    with env.begin(db=db) as txn:
        return txn.get(sha256(s.encode()))

def check_skip(filename):
    return (get_s(filename, skip_db) is not None)

def put_skip(filename):
    with env.begin(db=skip_db, write=True) as txn:
        txn.put(sha256(filename.encode()), b'1')

def check_fn(filename):
    return (get_s(filename, fn_db) is not None)

def get_fix_idx_vector(idx):
    return np.frombuffer(get_fix_idx(idx, b'v'), np.float32).reshape((1,512))

def get_fix_idx_filename(idx):
    return get_fix_idx(idx, b'n').decode()

def get_fix_idx(idx, postfix):
    idx = i2b(idx, postfix)
    with env.begin(db=fix_idx_db) as txn:
        return txn.get(idx)

def get_idx(faiss_index):
    with env.begin(db=idx_db) as txn:
        return b2i(txn.get(i2b(faiss_index)))

def get_idx_face(faiss_index):
    with env.begin(db=idx_face_db) as txn:
        res = txn.get(i2b(faiss_index))
        return (b2i(res), b2s(res[key_len()+1:]))

def put_idx(faiss_index, fix_idx):
    with env.begin(db=idx_db, write=True) as txn:
        txn.put(i2b(faiss_index), fix_idx)

def put_idx_face(faiss_index, fix_idx, face_idx):
    with env.begin(db=idx_face_db, write=True) as txn:
        txn.put(i2b(faiss_index), fix_idx + b'f' + s2b(face_idx))

def get_fn_idx(filename):
    return get_s(filename, fn_db)

def check_face(filename):
    idx = get_fn_idx(filename)
    if idx is None:
        return False
    with env.begin(db=fix_idx_db) as txn:
        return (txn.get(idx + b'f') is not None)

def sha256(buf):
    m = hashlib.sha256()
    m.update(buf)
    return m.digest()

def encode_face(annotation):
    f = annotation
    buf = struct.pack('<HHHHfHHHHHHHHHH', f['bbox'][0], f['bbox'][1], f['bbox'][2], f['bbox'][3], f['score'], f['landmarks'][0][0], f['landmarks'][0][1], f['landmarks'][1][0], f['landmarks'][1][1], f['landmarks'][2][0], f['landmarks'][2][1], f['landmarks'][3][0], f['landmarks'][3][1], f['landmarks'][4][0], f['landmarks'][4][1]) + f['embedding'].reshape((1,512)).astype('float32').tobytes()
    return buf

def decode_face(raw_face):
    annotation = {'bbox' : [0, 0, 0, 0], 'score': -1, 'landmarks': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 'embedding': np.zeros((1, 512))}
    f = annotation
    f['bbox'][0], f['bbox'][1], f['bbox'][2], f['bbox'][3], f['score'], f['landmarks'][0][0], f['landmarks'][0][1], f['landmarks'][1][0], f['landmarks'][1][1], f['landmarks'][2][0], f['landmarks'][2][1], f['landmarks'][3][0], f['landmarks'][3][1], f['landmarks'][4][0], f['landmarks'][4][1] = struct.unpack('<HHHHfHHHHHHHHHH', raw_face[0:32])
    f['embedding'] = np.frombuffer(raw_face[32:2080], np.float32).reshape((1,512))
    return annotation

def get_faces(idx):
    with env.begin(db=fix_idx_db) as txn:
        res = txn.get(idx + b'f')
        if res is None:
            return []
        else:
            annotations = []
            for i in range(b2i(res)):
                raw_face = txn.get(idx + b'f' + s2b(i))
                annotations.append(decode_face(raw_face))
            return annotations

def put_faces(idx, annotations):
    with env.begin(db=fix_idx_db, write=True) as txn:
        txn.put(idx + b'f', i2b(len(annotations)))
        for i, annotation in enumerate(annotations):
            txn.put(idx + b'f' + s2b(i), encode_face(annotation))

def put_fn(filename, vector):
    fn = filename.encode()
    fn_hash = sha256(fn)
    vector = vector.reshape((1,512)).astype('float32').tobytes()
    idx = None
    added = False
    with env.begin(db=fn_db) as txn:
        idx = txn.get(fn_hash)
    with env.begin(db=fix_idx_db, write=True) as txn:
        if idx is None:
            idx = get_next_idx()
            idx = txn.get(b'next')
            txn.put(idx + b'n', fn)
            txn.put(b'next', i2b(b2i(idx) + 1))
            added = True
        txn.put(idx + b'v', vector)
    with env.begin(db=fn_db, write=True) as txn:
        txn.put(fn_hash, idx)
    return idx
