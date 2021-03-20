from pathlib import Path
import atexit
import lmdb
import numpy as np
import struct
import faiss

from numpack import *
from db_config import config_map_size
import database

class ConfigDB:
    def __init__(self, path_prefix=None):
        if path_prefix is None:
            path_prefix = Path('.')
        self.path_prefix = path_prefix

        # LMDB environment
        self.env = None

        # Information about face tags
        self.tags_db = None
        self.tag_name_db = None
        self.cluster_db = None

        # Settings
        self.settings_db = None

        # Tag index and mapping
        self.index = None
        self.tag_map = {}
        self.image_map = {}
        self.tag_list = []

        self.cluster_map = {}

    # Open database
    def open_db(self):
        self.path = self.path_prefix / 'config.lmdb'
        self.env = lmdb.open(str(self.path), map_size=config_map_size, max_dbs=4)
        self.tag_name_db = self.env.open_db(b'tag_name_db')
        self.tags_db = self.env.open_db(b'tags_db', dupsort=True)
        self.settings_db = self.env.open_db(b'settings_db')
        self.cluster_db = self.env.open_db(b'cluster_db', dupsort=True)
        self.load_tags()

    def close(self):
        self.env.close()

    def load_tags(self):
        self.index = faiss.IndexFlatIP(512)
        self.cluster_map = {}
        self.tag_map = {}
        self.image_map = {}
        self.tag_list = []
        embeddings = []
        with self.env.begin() as txn:
            cursor = txn.cursor(self.tag_name_db)
            if cursor.first():
                for name, _ in cursor:
                    tag_name = name.decode()
                    self.tag_map[tag_name] = []
                    tag_cursor = txn.cursor(self.tags_db)
                    if tag_cursor.set_key(name):
                        for name_key, tag_data in tag_cursor:
                            if name_key != name:
                                break
                            fix_idx = database.i2b(b2i(tag_data))
                            face_idx = tag_data[8:10]
                            # FIXME: Use database instance that matches our path_prefix!
                            annotation = database.get_face(fix_idx, face_idx)
                            embedding = annotation['embedding'].reshape((512,)).astype('float32')
                            self.tag_map[tag_name].append((fix_idx, face_idx, len(self.tag_list), embedding))
                            if not fix_idx in self.image_map:
                                self.image_map[fix_idx] = {}
                            self.image_map[fix_idx][tag_name] = True
                            self.tag_list.append(tag_name)
                            embeddings.append(embedding)
        if len(embeddings) > 0:
            embeddings = np.array(embeddings)
            if not self.index.is_trained:
                self.index.train(embeddings)
            self.index.add(embeddings)

    # Cluster functions
    def add_cluster_tag(self, name, fix_idx, face_idx):
        face_key = database.i2b(fix_idx) + b'f' + s2b(face_idx)
        cluster_key = b'm' + name.encode()
        with self.env.begin(db=self.cluster_db) as txn:
            cursor = txn.cursor()
            if cursor.set_key_dup(cluster_key, face_key) and cursor.set_key_dup(b'f' + face_key, cluster_key):
                return True
        with self.env.begin(db=self.cluster_db, write=True) as txn:
            res = txn.get(b'f' + face_key)
            if res is None or txn.get(b'x' + face_key) is not None:
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

    def del_cluster_tag(self, name, fix_idx, face_idx, prevent_recluster=False):
        face_key = database.i2b(fix_idx) + b'f' + s2b(face_idx)
        cluster_key = b'm' + name.encode()
        with self.env.begin(db=self.cluster_db) as txn:
            cursor = txn.cursor()
            if not cursor.set_key_dup(cluster_key, face_key) and not cursor.set_key_dup(b'f' + face_key, cluster_key):
                return True
        with self.env.begin(db=self.cluster_db, write=True) as txn:
            res = txn.get(b'f' + face_key + b'o')
            txn.delete(cluster_key, face_key)
            txn.delete(b'f' + face_key, cluster_key)
            if res is None:
                return True
            txn.put(res, face_key)
            txn.put(b'f' + face_key, res)
            txn.delete(b'f' + face_key + b'o')
            if prevent_recluster:
                txn.put(b'x' + face_key)
        return True

    def purge_cluster_tag(self, name, fix_idx, face_idx, prevent_recluster):
        face_key = database.i2b(fix_idx) + b'f' + s2b(face_idx)
        cluster_key = b'm' + name.encode()
        with self.env.begin(db=self.cluster_db) as txn:
            target = txn.get(b'f' + face_key + b'o')
            if target is None:
                return False
            cursor = txn.cursor()
            cursor.set_key(cluster_key)
            for iter_key, value in cursor:
                if iter_key != cluster_key:
                    break
                res = txn.get(b'f' + value + b'o')
                if res == target:
                    self.del_cluster_tag(name, database.b2i(value), b2s(value[-2:]), prevent_recluster)
            return True

    def list_unnamed_clusters(self):
        clusters = []
        with self.env.begin(db=self.cluster_db) as txn:
            cursor = txn.cursor()
            if  cursor.set_range(b'c'):
                while True:
                    item = cursor.item()
                    if item is None or item[0][0] != b'c'[0]:
                        break
                    c_count = cursor.count()
                    if c_count > 1:
                        clusters.append((database.b2i(item[0][1:]), c_count))
                    if not cursor.next_nodup():
                        break
        return sorted(clusters, key=lambda x: x[0])

    def get_unnamed_cluster_contents(self, cluster_id):
        with self.env.begin(db=self.cluster_db) as txn:
            cursor = txn.cursor()
            cluster_key = b'c' + database.i2b(cluster_id)
            if not cursor.set_key(cluster_key):
                return None
            results = []
            for iter_key, value in cursor:
                if iter_key != cluster_key:
                    break
                results.append([(database.b2i(value), b2s(value[-2:])), 1.0])
            return results

    # Tag index functions
    def list_tags(self, cluster_mode):
        results = []
        if cluster_mode:
            with self.env.begin(db=self.cluster_db) as txn:
                cursor = txn.cursor()
                res = cursor.set_range(b'm')
                while res:
                    name, _ = cursor.item()
                    if name[0] != b'm'[0]:
                        break
                    num = cursor.count()
                    res = cursor.next_nodup()
                    results.append((num, name[1:].decode()))
            return results
        with self.env.begin() as txn:
            cursor = txn.cursor(self.tag_name_db)
            if cursor.first():
                for name, _ in cursor:
                    tag_name = name.decode()
                    tag_cursor = txn.cursor(tags_db)
                    tag_num = 0
                    if tag_cursor.set_key(name):
                        tag_num = tag_cursor.count()
                    results.append((tag_num, tag_name))
        return results

    def add_tag(self, name, fix_idx, face_idx, cluster_mode):
        if name == "":
            return False
        try:
            if cluster_mode:
                return self.add_cluster_tag(name, fix_idx, face_idx)
            with self.env.begin(db=self.tags_db, write=True) as txn:
                cursor = txn.cursor()
                annotation_key = database.i2b(fix_idx)
                face_key = s2b(face_idx)
                # FIXME: Use database instance that matches our path_prefix!
                embedding = database.get_face(annotation_key, face_key)['embedding'].reshape((1, 512)).astype('float32')
                key = name.encode()
                value = i2b(fix_idx) + face_key
                if not cursor.set_key_dup(key, value):
                    txn.put(key, value)
                    if name not in self.tag_map:
                        self.tag_map[name] = []
                    self.tag_map[name].append((database.i2b(fix_idx), face_key, len(self.tag_list), embedding.reshape((512,))))
                    if not fix_idx in self.image_map:
                        self.image_map[fix_idx] = {}
                    if annotation_key not in self.image_map:
                        self.image_map[annotation_key] = {}
                    self.image_map[annotation_key][name] = True
                    if name in self.cluster_map:
                        del self.cluster_map[name]
                    self.tag_list.append(name)
                    self.index.add(embedding)
            with self.env.begin(db=self.tag_name_db, write=True) as txn:
                txn.put(name.encode(), b'1')
            return True
        except:
            return False

    def has_tag(self, name, fix_idx, face_id, cluster_mode):
        fix_idx = database.i2b(fix_idx)
        if cluster_mode:
            if face_id is None: 
                return False
            face_key = fix_idx + b'f' + s2b(face_id)
            cluster_key = b'm' + name.encode()
            with self.env.begin(db=self.cluster_db) as txn:
                cursor = txn.cursor()
                if cursor.set_key_dup(cluster_key, face_key) and cursor.set_key_dup(b'f' + face_key, cluster_key):
                    return True
        return fix_idx in self.image_map and name in self.image_map[fix_idx]

    def get_face_tag(self, annotation, face_threshold, cluster_mode):
        if cluster_mode:
            with self.env.begin(db=self.cluster_db) as txn:
                cluster = txn.get(b'f' + annotation['face_key'])
                if cluster is None or cluster[0] != b'm'[0]:
                    return ""
                return cluster[1:].decode()
        embedding = annotation['embedding']
        D, I = self.index.search(embedding.reshape((1, 512)).astype('float32'), 1)
        if len(I[0]) < 1 or I[0][0] < 0 or D[0][0] < face_threshold:
            return ""
        return self.tag_list[I[0][0]]

    def get_tag_contents(self, name, cluster_mode):
        if cluster_mode:
            with self.env.begin(db=self.cluster_db) as txn:
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
        if name not in self.tag_map:
            return None
        results = []
        for fix_idx, face_idx, _, _ in self.tag_map[name]:
            results.append([(database.b2i(fix_idx), b2s(face_idx)), 1.0])
        return results

    def get_tag_embeddings(self, name, cluster_mode):
        if cluster_mode:
            cluster_items = self.get_tag_contents(name, True)
            if cluster_items is None:
                return None
            embeddings = []
            for cluster_item in cluster_items:
                fix_idx = database.i2b(cluster_item[0][0])
                face_id = s2b(cluster_item[0][1])
                # FIXME: Use database instance that matches our path_prefix!
                embeddings.append(database.get_face(fix_idx, face_id)['embedding'].reshape((512,)))
            return np.array(embeddings)
        if name not in self.tag_map or len(self.tag_map[name]) < 1:
            return None
        embeddings = []
        for _, _, _, embedding in self.tag_map[name]:
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        n_emb = embeddings.shape[0]
        if False and n_emb > 78*3: # FIXME: Disabled due to bugs
            kmeans = faiss.Kmeans(512, min(n_emb // 39, 150), niter=10, verbose=False)
            if name in self.cluster_map:
                kmeans = self.cluster_map[name]
            else:
                kmeans.train(embeddings)
                self.cluster_map[name] = kmeans
            if kmeans.centroids.shape[0] > 5:
                return kmeans.centroids
            else:
                return np.append(embeddings[1::3], kmeans.centroids, axis=0)
        return np.array(embeddings)

    def del_tag(self, name, fix_idx, face_idx, cluster_mode):
        if cluster_mode:
            return self.del_cluster_tag(name, fix_idx, face_idx)
        res = False
        with self.env.begin(db=self.tags_db, write=True) as txn:
            face_key = s2b(face_idx)
            key = name.encode()
            value = i2b(fix_idx) + face_key
            res = txn.delete(key, value=value)
        self.load_tags() # TODO: Replace this if this ever gets too slow
        return res

    # Settings functions
    def set_setting(self, name, value, conv):
        with self.env.begin(db=self.settings_db, write=True) as txn:
            txn.put(name.encode(), conv(value))

    def get_setting(self, name, default, conv):
        try:
            with self.env.begin(db=self.settings_db) as txn:
                return conv(txn.get(name.encode()))
        except:
            return default

    def set_setting_int(self, name, value):
        self.set_setting(name, value, i2b)

    def get_setting_int(self, name, default):
        return self.get_setting(name, default, b2i)

    def set_setting_bool(self, name, value):
        if value:
            self.set_setting(name, 1, i2b)
        else:
            self.set_setting(name, 0, i2b)

    def get_setting_bool(self, name, default):
        if default:
            default = 1
        else:
            default = 0
        return self.get_setting(name, default, b2i) == 1

    def set_setting_float(self, name, value):
        self.set_setting(name, value, f2b)

    def get_setting_float(self, name, default):
        return self.get_setting(name, default, b2f)


# Compatibility

default_config_db = None
default_attr_keys = [
    "env", "tags_db", "tag_name_db", "cluster_db", "settings_db",
    "index", "tag_map", "image_map", "tag_list", "cluster_map",
]
default_def_keys = [
    "load_tags",
    "add_cluster_tag", "del_cluster_tag", "purge_cluster_tag",
    "list_unnamed_clusters", "get_unnamed_cluster_contents",
    "list_tags", "add_tag", "has_tag", "del_tag",
    "get_face_tag", "get_tag_contents", "get_tag_embeddings",
    "set_setting", "get_setting",
    "set_setting_int", "get_setting_int",
    "set_setting_bool", "get_setting_bool",
    "set_setting_float", "get_setting_float",
]

def open_db():
    global default_config_db
    if default_config_db is not None:
        raise RuntimeError("default ConfigDB already opened")
    default_config_db = ConfigDB()
    default_config_db.open_db()
    atexit.register(lambda: default_config_db.close())

def __getattr__(name):
    if name not in default_attr_keys and name not in default_def_keys:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # Faking default error message...
    if default_config_db is None:
        # query-index.py uses this to check before opening the DB...
        if name == "env":
            return None
        raise AttributeError(f"module {__name__!r} default ConfigDB not opened, yet")
    #
    # (This doesn't seem to be necessary, as getattr() doesn't expose
    # the raw defs it seems, but cooked versions that have the self argument
    # already applied (or lambdas to that effect).)
    #if name in default_def_keys:
    #    return lambda *kv: getattr(default_config_db, name)(default_config_db, *kv)
    #
    return getattr(default_config_db, name)
