import numpy as np
import database
import config
from numpack import *
import faiss
import copy
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import DBSCAN

threshold = 0.33

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 0.000000001: 
       return v
    return v / norm

def cosine_distance(a, b):
    return 1. - (a @ b.T)

database.open_db()
config.open_db()

images = 0
faces = 0
index_map = []
with database.env.begin(db=database.fix_idx_db) as txn:
    images = database.b2i(txn.get(b'next'))
    for i in range(images):
        image_faces = b2s(txn.get(database.i2b(i) + b'f'))
        faces += image_faces
    print(f"Found {faces} faces")
    idx = 0
    arr = np.zeros((faces, 512)).astype('float32')
    for i in range(images):
        image_faces = b2s(txn.get(database.i2b(i) + b'f'))
        for j in range(image_faces):
            face_key = database.i2b(i) + b'f' + s2b(j)
            face = database.decode_face(txn.get(face_key))
            embedding = face['embedding'].reshape((512,))
            arr[idx] = embedding
            idx += 1
            index_map.append((i, j, face_key))
        #if idx > 200:
        #    break
    print(f"Filled matrix")
    #clusters = hcluster.fclusterdata(arr[0:idx], threshold, criterion='distance', metric='cosine', method='single')
    clusters = DBSCAN(eps=threshold, min_samples=1, metric=cosine_distance, algorithm='ball_tree', n_jobs=-1).fit(arr[0:idx]).labels_
    print("Calculated clusters")
    cluster_map = {}
    for i in range(idx):
        c = clusters[i]
        if c not in cluster_map:
            cluster_map[c] = []
        cluster_map[c].append(i)
    print("Assigning clusters...")

    #for i, cluster in enumerate(sorted(cluster_map.values(), key=lambda x: len(x))):
    #    print(f"Cluster {i} [{len(cluster)}]: ", end="")
    #    for idx in cluster:
    #        face = index_map[idx]
    #        print(face[0:2], end=", ")
    #    print()

    with config.env.begin(db=config.cluster_db, write=True) as c_txn:
        cluster_id = c_txn.get(b'next')
        if cluster_id is None:
            cluster_id = 0
        else:
            cluster_id = database.b2i(cluster_id)
        for i, cluster in enumerate(sorted(cluster_map.values(), key=lambda x: len(x))):
            cluster_faces = []
            c_key = b'c' + database.i2b(cluster_id)
            put_any = False
            for idx in cluster:
                face = index_map[idx]
                face_key = face[2]
                res = txn.get(b'f' + face_key)
                if res is None or res[0] == b'c'[0]:
                    c_txn.delete(b'f' + face_key)
                    c_txn.delete(b'f' + face_key + b'o')
                    c_txn.put(b'f' + face_key, c_key)
                    cluster_faces.append(face_key)
                    put_any = True
                c_txn.delete(c_key)
            for cluster_face in cluster_faces:
                c_txn.put(c_key, cluster_face)
                put_any = True
            if put_any:
                cluster_id += 1
        c_txn.delete(b'next')
        c_txn.put(b'next', database.i2b(cluster_id))
    print("Done.")
