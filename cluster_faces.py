import sys
import numpy as np
import database
import config
from numpack import *
import faiss
import copy
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import DBSCAN

# Clustering method to use from ['chinese_whispers', 'dbscan', 'hierarchical']. 'chinese_whispers' requires dlib (can be installed with pip).
cluster_method = 'hierarchical'

# Only show clusters, do not write them to database
show_clusters = False

# Drop cluster database and quit (WARNING: deletes all cluster tags!)
drop_clusters = False

if cluster_method == 'chinese_whispers':
    try:
        import dlib
    except:
        print("Error: Failed to load dlib. Please install it using 'pip install dlib'.")
        sys.exit(1)

threshold_cosine = 0.33
threshold_euclidean = np.sqrt(2 * threshold_cosine)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 0.000000001: 
       return v
    return v / norm

database.open_db()
config.open_db()

if drop_clusters:
    with config.env.begin(write=True) as txn:
        txn.drop(db=config.cluster_db)
    print("Deleted clusters and cluster tags.")
    sys.exit(0)

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
    if cluster_method == 'chinese_whispers':
        arr = []
    else:
        arr = np.zeros((faces, 512)).astype('float32')
    with config.env.begin(db=config.cluster_db, write=True) as c_txn:
        for i in range(images):
            image_faces = b2s(txn.get(database.i2b(i) + b'f'))
            for j in range(image_faces):
                face_key = database.i2b(i) + b'f' + s2b(j)
                if c_txn.get(b'x' + face_key) is not None:
                    faces -= 1
                    continue
                face = database.decode_face(txn.get(face_key))
                embedding = face['embedding'].reshape((512,))
                if cluster_method == 'chinese_whispers':
                    arr.append(dlib.vector(list(embedding)))
                else:
                    arr[idx] = embedding
                idx += 1
                index_map.append((i, j, face_key))
            #if idx > 200:
            #    break
    print(f"Filled matrix of {faces} faces")

    if cluster_method == 'chinese_whispers':
        clusters = dlib.chinese_whispers_clustering(arr, float(threshold_euclidean))
    elif cluster_method == 'dbscan':
        clusters = DBSCAN(eps=threshold_euclidean, min_samples=5, n_jobs=8, algorithm='ball_tree').fit(arr[0:idx]).labels_
    elif cluster_method == 'hierarchical':
        clusters = hcluster.fclusterdata(arr[0:idx], threshold_cosine, criterion='distance', metric='cosine', method='single')
    else:
        print("Error: Unknown clustering method.")
        sys.exit(1)
    print("Calculated clusters")

    cluster_map = {}
    for i in range(idx):
        c = clusters[i]
        if c < 0:
            continue
        if c not in cluster_map:
            cluster_map[c] = []
        cluster_map[c].append(i)

    if show_clusters:
        for i, cluster in enumerate(sorted(cluster_map.values(), key=lambda x: len(x))):
            print(f"Cluster {i} [{len(cluster)}]: ", end="")
            for idx in cluster:
                face = index_map[idx]
                print(face[0:2], end=", ")
            print()
        sys.exit(0)

    print("Assigning clusters...")
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
