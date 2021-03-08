import atexit
import lmdb
import numpy as np
import struct
import faiss

from numpack import *
import database

# LMDB environment
env = None

# Information about face tags
tags_db = None
tag_name_db = None
cluster_db = None

# Settings
settings_db = None

# Tag index and mapping
index = None
tag_map = {}
tag_list = []

# Open database
def open_db(map_size=1024*1024*1024):
    global env, tags_db, tag_name_db, settings_db, cluster_db
    env = lmdb.open('config.lmdb', map_size=map_size, max_dbs=4)
    tag_name_db = env.open_db(b'tag_name_db')
    tags_db = env.open_db(b'tags_db', dupsort=True)
    settings_db = env.open_db(b'settings_db')
    cluster_db = env.open_db(b'cluster_db', dupsort=True)
    load_tags()
    atexit.register(lambda: env.close())

def load_tags():
    global index, tag_map, tag_list
    index = faiss.IndexFlatIP(512)
    tag_map = {}
    tag_list = []
    embeddings = []
    with env.begin() as txn:
        cursor = txn.cursor(tag_name_db)
        if cursor.first():
            for name, _ in cursor:
                tag_name = name.decode()
                tag_map[tag_name] = []
                tag_cursor = txn.cursor(tags_db)
                if tag_cursor.set_key(name):
                    for name_key, tag_data in tag_cursor:
                        if name_key != name:
                            break
                        fix_idx = database.i2b(b2i(tag_data))
                        face_idx = tag_data[8:10]
                        annotation = database.get_face(fix_idx, face_idx)
                        embedding = annotation['embedding'].reshape((512,)).astype('float32')
                        tag_map[tag_name].append((fix_idx, face_idx, len(tag_list), embedding))
                        tag_list.append(tag_name)
                        embeddings.append(embedding)
    if len(embeddings) > 0:
        embeddings = np.array(embeddings)
        if not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)

# Cluster functions
def add_cluster_tag(name, fix_idx, face_idx):
    face_key = database.i2b(fix_idx) + b'f' + s2b(face_idx)
    cluster_key = b'm' + name.encode()
    with env.begin(db=cluster_db) as txn:
        cursor = txn.cursor()
        if cursor.set_key_dup(cluster_key, face_key) and cursor.set_key_dup(b'f' + face_key, cluster_key):
            return True
    with env.begin(db=cluster_db, write=True) as txn:
        res = txn.get(b'f' + face_key)
        if res is None:
            txn.put(cluster_key, face_key)
            txn.put(b'f' + face_key, cluster_key)
            return True
        txn.delete(b'f' + face_key)
        if res[0] == b'm'[0]:
            txn.delete(res, face_key)
            txn.put(cluster_key, face_key)
            txn.put(b'f' + face_key, cluster_key)
            return True
        if res[0] == b'c'[0]:
            cursor = txn.cursor()
            cursor.set_key(res)
            for iter_key, value in cursor:
                if iter_key != res:
                    break
                txn.delete(b'f' + value)
                txn.delete(b'f' + value + b'o')
                txn.delete(res, value)
                txn.put(b'f' + value, cluster_key)
                txn.put(b'f' + value + b'o', res)
                txn.put(cluster_key, value)
            return True
    return False

def del_cluster_tag(name, fix_idx, face_idx):
    face_key = database.i2b(fix_idx) + b'f' + s2b(face_idx)
    cluster_key = b'm' + name.encode()
    with env.begin(db=cluster_db) as txn:
        cursor = txn.cursor()
        if not cursor.set_key_dup(cluster_key, face_key) and not cursor.set_key_dup(b'f' + face_key, cluster_key):
            return True
    with env.begin(db=cluster_db, write=True) as txn:
        res = txn.get(b'f' + face_key + b'o')
        txn.delete(cluster_key, face_key)
        txn.delete(b'f' + face_key, cluster_key)
        if res is None:
            return True
        txn.put(res, face_key)
        txn.put(b'f' + face_key, res)
        txn.delete(b'f' + face_key + b'o')
    return True

def purge_cluster_tag(name, fix_idx, face_idx):
    face_key = database.i2b(fix_idx) + b'f' + s2b(face_idx)
    cluster_key = b'm' + name.encode()
    with env.begin(db=cluster_db) as txn:
        res = txn.get(b'f' + face_key + b'o')
        if res is None:
            return False
        cursor = txn.cursor()
        cursor.set_key(cluster_key)
        for iter_key, value in cursor:
            if iter_key != cluster_key:
                break
            del_cluster_tag(name, database.b2i(value), b2s(value[-2:]))
        return True

# Tag index functions
def add_tag(name, fix_idx, face_idx, cluster_mode):
    if name == "":
        return False
    try:
        if cluster_mode:
            return add_cluster_tag(name, fix_idx, face_idx)
        with env.begin(db=tags_db, write=True) as txn:
            cursor = txn.cursor()
            annotation_key = database.i2b(fix_idx)
            face_key = s2b(face_idx)
            embedding = database.get_face(annotation_key, face_key)['embedding'].reshape((1, 512)).astype('float32')
            key = name.encode()
            value = i2b(fix_idx) + face_key
            if not cursor.set_key_dup(key, value):
                txn.put(key, value)
                if name not in tag_map:
                    tag_map[name] = []
                tag_map[name].append((database.i2b(fix_idx), face_key, len(tag_list), embedding.reshape((512,))))
                tag_list.append(name)
                index.add(embedding)
        with env.begin(db=tag_name_db, write=True) as txn:
            txn.put(name.encode(), b'1')
        return True
    except:
        return False

def get_face_tag(annotation, face_threshold, cluster_mode):
    if cluster_mode:
        with env.begin(db=cluster_db) as txn:
            cluster = txn.get(b'f' + annotation['face_key'])
            if cluster is None or cluster[0] != b'm'[0]:
                return ""
            return cluster[1:].decode()
    embedding = annotation['embedding']
    D, I = index.search(embedding.reshape((1, 512)).astype('float32'), 1)
    if len(I[0]) < 1 or I[0][0] < 0 or D[0][0] < face_threshold:
        return ""
    return tag_list[I[0][0]]

def get_tag_contents(name, cluster_mode):
    if cluster_mode:
        with env.begin(db=cluster_db) as txn:
            cursor = txn.cursor()
            cluster_key = b'm' + name.encode()
            if not cursor.set_key(cluster_key):
                return None
            results = []
            for iter_key, value in cursor:
                if iter_key != cluster_key:
                    break
                results.append([(database.b2i(value), b2s(value[-2:])), 1.0])
            return results
    if name not in tag_map:
        return None
    results = []
    for fix_idx, face_idx, _, _ in tag_map[name]:
        results.append([(database.b2i(fix_idx), b2s(face_idx)), 1.0])
    return results

def get_tag_embeddings(name, cluster_mode):
    if cluster_mode:
        cluster_items = get_tag_contents(name, True)
        embeddings = []
        for cluster_item in cluster_items:
            fix_idx = database.i2b(cluster_item[0][0])
            face_id = s2b(cluster_item[0][1])
            embeddings.append(database.get_face(fix_idx, face_id)['embedding'].reshape((512,)))
        return np.array(embeddings)
    if name not in tag_map or len(tag_map[name]) < 1:
        return None
    embeddings = []
    for _, _, _, embedding in tag_map[name]:
        embeddings.append(embedding)
    return np.array(embeddings)

def del_tag(name, fix_idx, face_idx, cluster_mode):
    if cluster_mode:
        return del_cluster_tag(name, fix_idx, face_idx)
    res = False
    with env.begin(db=tags_db, write=True) as txn:
        face_key = s2b(face_idx)
        key = name.encode()
        value = i2b(fix_idx) + face_key
        res = txn.delete(key, value=value)
    load_tags() # TODO: Replace this if this ever gets too slow
    return res

# Settings functions
def set_setting(name, value, conv):
    with env.begin(db=settings_db, write=True) as txn:
        txn.put(name.encode(), conv(value))

def get_setting(name, default, conv):
    try:
        with env.begin(db=settings_db) as txn:
            return conv(txn.get(name.encode()))
    except:
        return default

def set_setting_int(name, value):
    set_setting(name, value, i2b)

def get_setting_int(name, default):
    return get_setting(name, default, b2i)

def set_setting_bool(name, value):
    if value:
        set_setting(name, 1, i2b)
    else:
        set_setting(name, 0, i2b)

def get_setting_bool(name, default):
    if default:
        default = 1
    else:
        default = 0
    return get_setting(name, default, b2i) == 1

def set_setting_float(name, value):
    set_setting(name, value, f2b)

def get_setting_float(name, default):
    return get_setting(name, default, b2f)
