import numpy as np
import database
from numpack import *
import faiss
import copy

# This is a very slow experiment, not useful probably

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 0.000000001: 
       return v
    return v / norm

database.open_db()

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
            face = database.decode_face(txn.get(database.i2b(i) + b'f' + s2b(j)))
            arr[idx] = face['embedding'].reshape((512,))
            idx += 1
            index_map.append((i, j))
        if idx > 400:
            break
    print(f"Filled matrix")
    in_arr = arr
    cluster_array = None
    for i in range(3):
        print(f"Iterating")
        faces_index = faiss.IndexFlatIP(512)
        #quantizer = faiss.IndexFlatIP(512)
        #faces_index = faiss.IndexIVFFlat(quantizer, 512, 32, faiss.METRIC_INNER_PRODUCT)
        #faces_index.nprobe = 4
        if not faces_index.is_trained:
            faces_index.train(in_arr)
        faces_index.add(in_arr)
        print(f"Filled index")
        cluster_array = copy.copy(in_arr)
        for i in range(idx):
            _, D, I = faces_index.range_search(x=in_arr[i].reshape((1,512)), thresh=0.65)
            I = np.array(I)
            I = I[I >= 0]
            embeddings = in_arr[I]
            cluster_array[i] = normalize(embeddings.mean(0, keepdims=True))
        print(f"Filled cluster heads")
        in_arr = cluster_array
    print("Determining clusters")
    faces_index = faiss.IndexFlatIP(512)
    #quantizer = faiss.IndexFlatIP(512)
    #faces_index = faiss.IndexIVFFlat(quantizer, 512, 32, faiss.METRIC_INNER_PRODUCT)
    #faces_index.nprobe = 4
    if not faces_index.is_trained:
        faces_index.train(cluster_array)
    faces_index.add(cluster_array)
    clusters = {}
    for i in range(idx):
        _, D, I = faces_index.range_search(x=arr[i].reshape((1,512)), thresh=0.65)
        if len(I) < 1 or I[0] < 0:
            continue
        I = np.array(I)
        I = I[I >= 0]
        if I[0] not in clusters:
            clusters[I[0]] = []
        clusters[I[0]].append(i)
    print("Calculated cluster belonging")
    for i, cluster in enumerate(sorted(clusters.values(), key=lambda x: len(x))):
        print("Cluster:")
        for idx in cluster:
            print(index_map[idx], end=", ")
        print()
