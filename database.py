import logging
from pathlib import Path
import weakref
import atexit
import lmdb
import numpy as np
import hashlib
import struct
from numpack import s2b, b2s
from db_config import vectors_map_size

logger = logging.getLogger(__name__)

by_path_prefix = weakref.WeakValueDictionary()

class DB:
    def __init__(self, *, path_prefix=None, pack_type=None):
        if path_prefix is None:
            path_prefix = Path('.')
        elif type(path_prefix) is str:
            path_prefix = Path(path_prefix)
        self.path_prefix = path_prefix
        by_path_prefix[str(self.path_prefix)] = self

        logger.debug("DB %#x: Instantiating DB(path_prefix=%r, pack_type=%r)", id(self), self.path_prefix, pack_type)

        # LMDB environment
        self.env = None

        # Filename hash to index
        self.fn_db = None

        # Filenames that were skipped
        self.skip_db = None

        # Fixed mapping from uint64 to filename and to vector (+'n' for filename, +'v' for vector, +'f' for face count, +'f'+fidx for face data)
        self.fix_idx_db = None

        # faiss index to fixed index
        self.idx_db = None

        # faiss face index to faiss index and face index in image
        self.idx_face_db = None

        # uint64
        self.fallback_int_type = '<Q'

        self.int_type = pack_type

    def open_db(self, *, pack_type=None):
        if pack_type is not None:
            self.int_type = pack_type
        self.path = self.path_prefix / 'vectors.lmdb'
        logger.info("DB %#x: Opening DB (pack_type=%r, resulting requested int_type=%r) at: %s", id(self),
            pack_type, self.int_type, self.path)
        self.env = lmdb.open(str(self.path), map_size=vectors_map_size, max_dbs=5)
        weakref.finalize(self, self.close)

        self.fn_db = self.env.open_db(b'fn_db')
        self.skip_db = self.env.open_db(b'skip_db')
        self.fix_idx_db = self.env.open_db(b'fix_idx_db')
        self.idx_db = self.env.open_db(b'idx_db')
        self.idx_face_db = self.env.open_db(b'idx_face_db')

        with self.env.begin(db=self.fix_idx_db, write=True) as txn:
            res = txn.get(b'int_type')
            if res is None:
                using_fallback = False
                if self.int_type is None:
                    using_fallback = True
                    self.int_type = self.fallback_int_type
                logger.info("DB %#x: ... initializing DB int_type to %r (using_fallback=%r)", id(self),
                    self.int_type, using_fallback)
                txn.put(b'int_type', self.int_type.encode('utf-8', 'surrogateescape'))
            else:
                db_int_type = res.decode()
                matched = None
                if self.int_type is not None:
                    matched = db_int_type == self.int_type
                    if not matched:
                        raise RuntimeError(f"database int_type={db_int_type!r} doesn't match requested int_type={self.int_type!r}")
                self.int_type = db_int_type
                logger.debug("DB %#x: ... loaded DB int_type of %r (matched=%r)", id(self), self.int_type, matched)
            res = txn.get(b'next')
            if res is None:
                txn.put(b'next', self.i2b(0))

    def close(self):
        if self.env is not None:
            logger.debug("DB %#x: Closing DB", id(self))
            self.env.close()
            self.env = None

    def key_len(self):
        l = 8
        if self.int_type == '<L':
            l = 4
        return l

    def i2b(self, i, postfix=None):
        packed = struct.pack(self.int_type, i)
        if postfix is not None:
            packed += postfix
        return packed

    def b2i(self, b):
        return struct.unpack(self.int_type, b[0:self.key_len()])[0]

    def get_next_idx(self):
        with self.env.begin(db=self.fix_idx_db) as txn:
            return txn.get(b'next')

    def get_s(self, s, db):
        with self.env.begin(db=db) as txn:
            return txn.get(self.sha256(s.encode('utf-8', 'surrogateescape')))

    def check_skip(self, filename):
        return (self.get_s(filename, self.skip_db) is not None)

    def put_skip(self, filename):
        with self.env.begin(db=self.skip_db, write=True) as txn:
            txn.put(self.sha256(filename.encode('utf-8', 'surrogateescape')), b'1')

    def check_fn(self, filename):
        return (self.get_s(filename, self.fn_db) is not None)

    def get_fix_idx_vector(self, idx):
        return np.frombuffer(self.get_fix_idx(idx, b'v'), np.float32).reshape((1,512))

    def get_fix_idx_filename(self, idx):
        return self.get_fix_idx(idx, b'n').decode('utf-8', 'surrogateescape')

    def get_fix_idx(self, idx, postfix):
        idx = self.i2b(idx, postfix)
        with self.env.begin(db=self.fix_idx_db) as txn:
            return txn.get(idx)

    def get_idx(self, faiss_index):
        with self.env.begin(db=self.idx_db) as txn:
            return self.b2i(txn.get(self.i2b(faiss_index)))

    def get_idx_face(self, faiss_index):
        with self.env.begin(db=self.idx_face_db) as txn:
            res = txn.get(self.i2b(faiss_index))
            return (self.b2i(res), b2s(res[self.key_len()+1:]))

    def put_idx(self, faiss_index, fix_idx):
        with self.env.begin(db=self.idx_db, write=True) as txn:
            txn.put(self.i2b(faiss_index), fix_idx)

    def put_idx_face(self, faiss_index, fix_idx, face_idx):
        with self.env.begin(db=self.idx_face_db, write=True) as txn:
            txn.put(self.i2b(faiss_index), fix_idx + b'f' + s2b(face_idx))

    def get_fn_idx(self, filename):
        return self.get_s(filename, self.fn_db)

    def check_face(self, filename):
        idx = self.get_fn_idx(filename)
        if idx is None:
            return False
        with self.env.begin(db=self.fix_idx_db) as txn:
            return (txn.get(idx + b'f') is not None)

    @classmethod
    def sha256(cls, buf):
        m = hashlib.sha256()
        m.update(buf)
        return m.digest()

    @classmethod
    def encode_face(cls, annotation):
        f = annotation
        buf = struct.pack('<hhhhfhhhhhhhhhh', f['bbox'][0], f['bbox'][1], f['bbox'][2], f['bbox'][3], f['score'], f['landmarks'][0][0], f['landmarks'][0][1], f['landmarks'][1][0], f['landmarks'][1][1], f['landmarks'][2][0], f['landmarks'][2][1], f['landmarks'][3][0], f['landmarks'][3][1], f['landmarks'][4][0], f['landmarks'][4][1]) + f['embedding'].reshape((1,512)).astype('float32').tobytes()
        return buf

    @classmethod
    def decode_face(cls, raw_face):
        annotation = {'bbox' : [0, 0, 0, 0], 'score': -1, 'landmarks': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 'embedding': np.zeros((1, 512))}
        f = annotation
        f['bbox'][0], f['bbox'][1], f['bbox'][2], f['bbox'][3], f['score'], f['landmarks'][0][0], f['landmarks'][0][1], f['landmarks'][1][0], f['landmarks'][1][1], f['landmarks'][2][0], f['landmarks'][2][1], f['landmarks'][3][0], f['landmarks'][3][1], f['landmarks'][4][0], f['landmarks'][4][1] = struct.unpack('<hhhhfhhhhhhhhhh', raw_face[0:32])
        f['embedding'] = np.frombuffer(raw_face[32:2080], np.float32).reshape((1,512))
        return annotation

    def get_face(self, idx, face_idx):
        with self.env.begin(db=self.fix_idx_db) as txn:
            face_key = idx + b'f' + face_idx
            logger.debug("DB %#x: get_face(idx=%r, face_idx=%r) -> face_key=%r", id(self), idx, face_idx, face_key)
            raw_face = txn.get(face_key)
            logger.debug("DB %#x: ... raw_face=%s", id(self),
                ('None' if raw_face is None else repr(raw_face[:16]) + '[...]'))
            annotation = self.decode_face(raw_face)
            annotation['face_key'] = face_key
            return annotation

    def get_faces(self, idx):
        with self.env.begin(db=self.fix_idx_db) as txn:
            res = txn.get(idx + b'f')
            if res is None:
                return []
            else:
                annotations = []
                for i in range(self.b2i(res)):
                    annotations.append(self.get_face(idx, s2b(i)))
                return annotations

    def put_faces(self, idx, annotations):
        with self.env.begin(db=self.fix_idx_db, write=True) as txn:
            txn.put(idx + b'f', self.i2b(len(annotations)))
            for i, annotation in enumerate(annotations):
                txn.put(idx + b'f' + s2b(i), self.encode_face(annotation))

    def put_fn(self, filename, vector):
        fn = filename.encode('utf-8', 'surrogateescape')
        fn_hash = self.sha256(fn)
        vector = vector.reshape((1,512)).astype('float32').tobytes()
        idx = None
        with self.env.begin(db=self.fn_db) as txn:
            idx = txn.get(fn_hash)
        with self.env.begin(db=self.fix_idx_db, write=True) as txn:
            if idx is None:
                idx = self.get_next_idx()
                idx = txn.get(b'next')
                txn.put(idx + b'n', fn)
                txn.put(b'next', self.i2b(self.b2i(idx) + 1))
                added = True
            txn.put(idx + b'v', vector)
        with self.env.begin(db=self.fn_db, write=True) as txn:
            txn.put(fn_hash, idx)
        return idx


# Compatibility

sha256 = DB.sha256
encode_face = DB.encode_face
decode_face = DB.decode_face

default_db = None
default_keys = [
    'env', 'fn_db', 'skip_db', 'fix_idx_db', 'idx_db', 'idx_face_db', 'int_type',
    'key_len', 'i2b', 'b2i',
    'get_next_idx', 'get_s', 'check_skip', 'put_skip', 'check_fn',
    'get_fix_idx_vector', 'get_fix_idx_filename', 'get_fix_idx',
    'get_idx', 'get_idx_face', 'put_idx', 'put_idx_face', 'get_fn_idx',
    'check_face', 'get_face', 'get_faces', 'put_faces', 'put_fn',
]

def open_db(pack_type=None):  # (Note: Allows positional argument, as it's compatibility code!)
    global default_db
    if default_db is not None:
        raise RuntimeError("default DB already opened")
    default_db = DB()
    default_db.open_db(pack_type=pack_type)
    atexit.register(lambda: default_db.close())

def __getattr__(name):
    if name not in default_keys:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # Faking default error message...
    if default_db is None:
        # query-index.py uses this to check before opening the DB...
        if name == "env":
            return None
        raise AttributeError(f"module {__name__!r} default DB not opened, yet")
    return getattr(default_db, name)
