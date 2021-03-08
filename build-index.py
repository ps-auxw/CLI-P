import os
import os.path
import sys
import numpy as np
import lmdb
import warnings
from torch_device import device
import torch
import clip
import faiss
from PIL import Image

import database
from faces import get_faces as get_face_embeddings, load_arcface

# Maximum size of (sparse) LMDB file
map_size = 1024*1024*1024*20

# Enable to run face detection and calculate face embeddings that can be used to search for faces
faces = True

# If you have more than 2^32 images or faces, set this to '<Q'
pack_type = '<L'

# Split up index into this many clusters, 100 seems like a good number, but having at the very least 36 * clusters images is recommended
clusters = 100

# Accepted file extensions (have to be readable as standard RGB images by pillow and opencv)
file_extensions = ['.jpg', '.jpeg', '.png']

# Paths containing these will be skipped during index creation
skip_paths = []

model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

if faces:
    load_arcface()

database.open_db(map_size, pack_type)

try:
    with torch.no_grad():
        for base_path in sys.argv[1:]:
            print(f"CLIPing {base_path}...")
            for fn in os.listdir(base_path):
                tfn = os.path.join(base_path, fn)
                ext = os.path.splitext(fn)
                if len(ext) < 2 or not ext[1].lower() in file_extensions:
                    continue
                if database.check_skip(tfn):
                    continue
                clip_done = database.check_fn(tfn)
                faces_done = not faces or database.check_face(tfn)
                if clip_done and faces_done:
                    continue
                image = None
                try:
                    image = Image.open(tfn).convert("RGB")
                    idx = None
                    if not faces_done:
                        rgb = np.array(image)
                    if not clip_done:
                        image = transform(image).unsqueeze(0).to(device)
                        image_features = model.encode_image(image)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        idx = database.put_fn(tfn, image_features.detach().cpu().numpy())
                    else:
                        idx = database.get_fn_idx(tfn)
                    if not faces_done:
                        annotations = get_face_embeddings(image=rgb)
                        database.put_faces(idx, annotations)
                    print(".", end="", flush=True)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except Exception as e:
                    print("#", end="", flush=True)
                    database.put_skip(tfn)
                    continue
            print(flush=True)
except KeyboardInterrupt:
    print(f"Interrupted!")

i = 0
faces_i = 0
with database.env.begin(db=database.fn_db) as fn_txn:
    n = fn_txn.stat()['entries']
    with database.env.begin(db=database.fix_idx_db) as txn:
        nd = n
        count = 0
        faces_count = 0
        need_training = True
        faces_need_training = True
        if nd > 32768:
            nd = 32768
        cursor = fn_txn.cursor()
        if cursor.first():
            if faces:
                faces_array = np.zeros((nd, 512))
            images = np.zeros((nd, 512))
            print(f"Preparing indexes...")
            quantizer = faiss.IndexFlatIP(512)
            index = faiss.IndexIVFFlat(quantizer, 512, clusters, faiss.METRIC_INNER_PRODUCT)
            if faces:
                faces_quantizer = faiss.IndexFlatIP(512)
                faces_index = faiss.IndexIVFFlat(faces_quantizer, 512, clusters, faiss.METRIC_INNER_PRODUCT)
            print(f"Generating matrix...")
            for fn_hash, fix_idx in cursor:
                fn = txn.get(fix_idx + b'n').decode()
                skip = False
                for skip_path in skip_paths:
                    if skip_path in fn:
                        skip = True
                        break
                if skip:
                    continue
                v = database.get_fix_idx_vector(database.b2i(fix_idx)).reshape((512,))
                images[count, :] = v
                count += 1
                database.put_idx(i, fix_idx)
                i += 1
                if count == nd:
                    count = 0
                    images = images.astype('float32')
                    if need_training:
                        print(f"Training index {images.shape}...")
                        index.train(images)
                        need_training = False
                    print(f"Adding to index...")
                    index.add(images)
                    images = np.zeros((nd, 512))
                if faces:
                    annotations = database.get_faces(fix_idx)
                    for face_idx, annotation in enumerate(annotations):
                        faces_array[faces_count, :] = annotation['embedding'][0].reshape((512,))
                        faces_count += 1
                        database.put_idx_face(faces_i, fix_idx, face_idx)
                        faces_i += 1
                        if faces_count == nd:
                            faces_count = 0
                            faces_array = faces_array.astype('float32')
                            if faces_need_training:
                                print(f"Training faces index {faces_array.shape}...")
                                faces_index.train(faces_array)
                                faces_need_training = False
                            print(f"Adding to faces index...")
                            faces_index.add(faces_array)
                            faces_array = np.zeros((nd, 512))
            if count > 0:
                images = images[0:count].astype('float32')
                if need_training:
                    print(f"Training index {images.shape}...")
                    index.train(images)
                print(f"Adding to index...")
                index.add(images)
            if faces_count > 0:
                faces_array = faces_array[0:faces_count].astype('float32')
                if faces_need_training:
                    print(f"Training faces index {faces_array.shape}...")
                    faces_index.train(faces_array)
                print(f"Adding to faces index...")
                faces_index.add(faces_array)
            print(f"Saving index...")
            faiss.write_index(index, "images.index")
            if faces:
                print(f"Saving faces index...")
                faiss.write_index(faces_index, "faces.index")

print(f"Indexed {i} images and {faces_i} faces.")
print(f"Done!")

database.env.close()
