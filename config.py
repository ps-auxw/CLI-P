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

# Settings
settings_db = None

# Tag index and mapping
index = None
tag_map = {}
image_map = {}
tag_list = []

cluster_map = {}

# Open database
def open_db(map_size=1024*1024*1024):
    global env, tags_db, tag_name_db, settings_db
    env = lmdb.open('config.lmdb', map_size=map_size, max_dbs=3)
    tag_name_db = env.open_db(b'tag_name_db')
    tags_db = env.open_db(b'tags_db', dupsort=True)
    settings_db = env.open_db(b'settings_db')
    load_tags()
    atexit.register(lambda: env.close())

def load_tags():
    global index, tag_map, tag_list, image_map, cluster_map
    index = faiss.IndexFlatIP(512)
    cluster_map = {}
    tag_map = {}
    image_map = {}
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
                        if not fix_idx in image_map:
                            image_map[fix_idx] = {}
                        image_map[fix_idx][tag_name] = True
                        tag_list.append(tag_name)
                        embeddings.append(embedding)
    if len(embeddings) > 0:
        embeddings = np.array(embeddings)
        if not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)

# Tag index functions
def list_tags():
    results = []
    with env.begin() as txn:
        cursor = txn.cursor(tag_name_db)
        if cursor.first():
            for name, _ in cursor:
                tag_name = name.decode()
                tag_cursor = txn.cursor(tags_db)
                tag_num = 0
                if tag_cursor.set_key(name):
                    tag_num = tag_cursor.count()
                results.append((tag_num, tag_name))
    return results

def add_tag(name, fix_idx, face_idx):
    global image_map, cluster_map
    if name == "":
        return False
    try:
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
                if not fix_idx in image_map:
                    image_map[fix_idx] = {}
                if annotation_key not in image_map:
                    image_map[annotation_key] = {}
                image_map[annotation_key][name] = True
                if name in cluster_map:
                    del cluster_map[name]
                tag_list.append(name)
                index.add(embedding)
        with env.begin(db=tag_name_db, write=True) as txn:
            txn.put(name.encode(), b'1')
        return True
    except:
        return False

def has_tag(name, fix_idx):
    fix_idx = database.i2b(fix_idx)
    return fix_idx in image_map and name in image_map[fix_idx]

def get_face_tag(annotation, face_threshold):
    embedding = annotation['embedding']
    D, I = index.search(embedding.reshape((1, 512)).astype('float32'), 1)
    if len(I[0]) < 1 or I[0][0] < 0 or D[0][0] < face_threshold:
        return ""
    return tag_list[I[0][0]]

def get_tag_contents(name):
    if name not in tag_map:
        return None
    results = []
    for fix_idx, face_idx, _, _ in tag_map[name]:
        results.append([(database.b2i(fix_idx), b2s(face_idx)), 1.0])
    return results

def get_tag_embeddings(name):
    if name not in tag_map or len(tag_map[name]) < 1:
        return None
    embeddings = []
    for _, _, _, embedding in tag_map[name]:
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    n_emb = embeddings.shape[0]
    if n_emb > 78*3:
        kmeans = faiss.Kmeans(512, min(n_emb // 39, 150), niter=10, verbose=False)
        if name in cluster_map:
            kmeans = cluster_map[name]
        else:
            kmeans.train(embeddings)
            cluster_map[name] = kmeans
        if kmeans.centroids.shape[0] > 50:
            return kmeans.centroids
        else:
            return np.append(embeddings[1::2], kmeans.centroids, axis=0)
    return np.array(embeddings)

def del_tag(name, fix_idx, face_idx):
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
