import sys
import random
import numpy as np
import database
import config
from numpack import *
import faiss
import copy

# Only show clusters, do not write them to database
show_clusters = False

# Drop cluster database and quit (WARNING: deletes all cluster tags!)
drop_clusters = False

# Only process first cut_off faces, for faster debugging. Set to 0 to disable.
cut_off = 0

threshold = 0.7
threshold_low = 0.5
threshold_avg_intra = 0.475

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 0.000000001: 
       return v
    return v / norm

database.open_db()
config.open_db()

index = faiss.read_index("faces.index")
index.nprobe = 16
candidates = 300

if drop_clusters:
    with config.env.begin(write=True) as txn:
        txn.drop(db=config.cluster_db)
    print("Deleted clusters and cluster tags.")
    sys.exit(0)

images = 0
faces = 0
index_map = []
faiss_map = {}
with database.env.begin(db=database.fix_idx_db) as txn:
    images = database.b2i(txn.get(b'next'))
    for i in range(images):
        image_faces = b2s(txn.get(database.i2b(i) + b'f'))
        faiss_map[i] = faces
        faces += image_faces
    print(f"Found {faces} faces")
    idx = 0

    arr = np.zeros((faces, candidates), np.int64) - 1
    no_merge = np.zeros((faces,), bool)
    face_set = []
    for i in range(images):
        image_faces = b2s(txn.get(database.i2b(i) + b'f'))
        face_set = []
        for j in range(image_faces):
            face_key = database.i2b(i) + b'f' + s2b(j)
            with config.env.begin(db=config.cluster_db) as c_txn:
                if c_txn.get(b'x' + face_key) is not None:
                    faces -= 1
                    continue
            face = database.decode_face(txn.get(face_key))
            embedding = face['embedding'].reshape((1,512))
            face_set.append((embedding, idx))
            _, D, I = index.range_search(x=embedding, thresh=threshold)
            random.shuffle(I)
            s_idx = 0
            for faiss_idx in I:
                if faiss_idx >= 0 and s_idx < candidates:
                    faiss_fix_idx, faiss_face_id = database.get_idx_face(faiss_idx)
                    arr[idx, s_idx] = faiss_map[faiss_fix_idx] + faiss_face_id
                    s_idx += 1
            idx += 1
            index_map.append((i, j, face_key))
            if idx % 100 == 0 and idx > 0:
                print(".", end="", flush=True)
        n_face_set = len(face_set)
        # If an image contains multiple faces and at least two are quite dissimilar (< threshold_low), but there is also a path between pairs of faces within the regular similarity threshold, the faces should not be used for merging clusters as they are usually some sort of artifact.
        if n_face_set > 2:
            embeddings = np.array(list(map(lambda x: x[0][0], face_set)))
            similarities = embeddings @ embeddings.T
            minimum = similarities.argmin()
            x, y = minimum // n_face_set, minimum % n_face_set
            if True or ((similarities > threshold) == (similarities < 0.999)).any() and (similarities < threshold_low).any() and x != y:
                valid = similarities > threshold
                queue = [x]
                done = {x: True}
                while queue:
                    node = queue.pop()
                    for k, validity in enumerate(valid[node]):
                        if validity:
                            if k == y:
                                # A path was found, excluse this image's faces from cluster merging
                                for face_set_item in face_set:
                                    no_merge[face_set_item[1]] = True
                                #print("--> ", i)
                                queue = []
                                break
                            elif k not in done:
                                done[k] = True
                                queue.append(k)
        if cut_off > 0 and idx > cut_off:
            break
    max_idx = idx
    arr = arr[0:max_idx]

    print(f"\nFilled matrix of {faces} faces")

    merge_list = {}
    clusters = np.zeros((max_idx,)).astype('int64') - 1
    cluster_id = 0
    for i in range(max_idx):
        # Skip cluster generation for faces marked for no cluster merging, but they may still get picked up by other clusters
        if no_merge[i]:
            if clusters[i] < 0:
                this_cluster = cluster_id
                cluster_id += 1
                clusters[i] = this_cluster
            continue

        found_clusters = []
        for j in range(candidates):
            other = arr[i, j]
            if cut_off > 0 and other > cut_off:
                continue
            if other < 0:
                break
            if not no_merge[other] and clusters[other] > -1:
                found_clusters.append(clusters[other])
        this_cluster = cluster_id
        cluster_id += 1
        processed = {}
        processed[this_cluster] = True
        if len(found_clusters) > 0:
            this_cluster = min(found_clusters)
            for found in found_clusters:
                while True:
                    if found in processed:
                        break
                    if found not in merge_list:
                        merge_list[found] = this_cluster
                        processed[found] = True
                        break
                    existing_merge = merge_list[found]
                    merge_list[found] = this_cluster
                    processed[found] = True
                    found = existing_merge
        clusters[i] = this_cluster
        for j in range(candidates):
            other = arr[i, j]
            if cut_off > 0 and other > cut_off:
                continue
            if other < 0:
                break
            clusters[other] = this_cluster
        if i % 100 == 0 and i > 0:
            print(".", end="", flush=True)

    print("\nAssigned base clusters, merging now")

    for i in range(max_idx):
        if clusters[i] in merge_list:
            clusters[i] = merge_list[clusters[i]]

    print("Calculated clusters")

    cluster_map = {}
    for i in range(idx):
        c = clusters[i]
        if c < 0:
            continue
        if c not in cluster_map:
            cluster_map[c] = []
        cluster_map[c].append(i)

    print("Filtering big clusters with low mean similarities (sampled)...")
    embeddings = np.array((768, 512))
    del_list = []
    for i, cluster in enumerate(cluster_map.keys()):
        c_id = cluster
        cluster = cluster_map[c_id]
        c_len = len(cluster)
        if c_len < 16:
            continue
        pick = np.arange(c_len)
        if c_len > 768:
            pick = np.random.choice(pick, 768)
            c_len = 768
        for i, p in enumerate(pick):
            face = index_map[cluster[pick]]
            embeddings[i, :] = database.get_face(database.i2b(face[0]), s2b(face[1]))['embedding'].reshape((512,))
        c_embeddings = embeddings[0:c_len]
        c_similarity = c_embeddings @ c_embeddings.T
        if c_similarity[c_similarity < 0.999].mean() < threshold_avg_intra:
            del_list.append(c_id)
    for del_cluster in del_list:
        del cluster_map[del_cluster]
    print(f"Deleted {len(del_list)} low mean similarity clusters")

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
