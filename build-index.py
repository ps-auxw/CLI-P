import os
import os.path
from pathlib import Path
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


class Scanner:

    def __init__(self, *, faces=faces, pack_type=pack_type, clusters=clusters,
        file_extensions=file_extensions, skip_paths=skip_paths, path_prefix=None):
        # Copy instance variables from keyword arguments defaulted to globals.
        self.faces = faces
        self.pack_type = pack_type
        self.clusters = clusters
        self.file_extensions = list(file_extensions)
        self.skip_paths = list(skip_paths)
        if path_prefix is None:
            path_prefix = Path('.')
        elif type(path_prefix) is str:
            path_prefix = Path(path_prefix)
        self.path_prefix = path_prefix

        self.model, self.transform = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()

        if self.faces:
            load_arcface()

        self.db = database.get(path_prefix=self.path_prefix, pack_type=self.pack_type)

    def clip_paths(self, base_paths):
        with torch.no_grad():
            for base_path in base_paths:
                print(f"CLIPing {base_path}...")
                for fn in os.listdir(base_path):
                    tfn = os.path.join(base_path, fn)
                    ext = os.path.splitext(fn)
                    if len(ext) < 2 or not ext[1].lower() in self.file_extensions:
                        continue
                    if self.db.check_skip(tfn):
                        continue
                    clip_done = self.db.check_fn(tfn)
                    faces_done = not self.faces or self.db.check_face(tfn)
                    if clip_done and faces_done:
                        continue
                    image = None
                    try:
                        image = Image.open(tfn).convert("RGB")
                        idx = None
                        if not faces_done:
                            rgb = np.array(image)
                        if not clip_done:
                            image = self.transform(image).unsqueeze(0).to(device)
                            image_features = self.model.encode_image(image)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            idx = self.db.put_fn(tfn, image_features.detach().cpu().numpy())
                        else:
                            idx = self.db.get_fn_idx(tfn)
                        if not faces_done:
                            annotations = get_face_embeddings(image=rgb)
                            self.db.put_faces(idx, annotations)
                        print(".", end="", flush=True)
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt()
                    except Exception as e:
                        print("#", end="", flush=True)
                        self.db.put_skip(tfn)
                        continue
                print(flush=True)

    def index_images(self):
        i = 0
        faces_i = 0
        with self.db.env.begin(db=self.db.fn_db) as fn_txn:
            n = fn_txn.stat()['entries']
            with self.db.env.begin(db=self.db.fix_idx_db) as txn:
                nd = n
                count = 0
                faces_count = 0
                need_training = True
                faces_need_training = True
                if nd > 32768:
                    nd = 32768
                cursor = fn_txn.cursor()
                if cursor.first():
                    if self.faces:
                        faces_array = np.zeros((nd, 512))
                    images = np.zeros((nd, 512))
                    print(f"Preparing indexes...")
                    quantizer = faiss.IndexFlatIP(512)
                    index = faiss.IndexIVFFlat(quantizer, 512, clusters, faiss.METRIC_INNER_PRODUCT)
                    if self.faces:
                        faces_quantizer = faiss.IndexFlatIP(512)
                        faces_index = faiss.IndexIVFFlat(faces_quantizer, 512, clusters, faiss.METRIC_INNER_PRODUCT)
                    print(f"Generating matrix...")
                    for fn_hash, fix_idx in cursor:
                        fn = txn.get(fix_idx + b'n').decode()
                        skip = False
                        for skip_path in self.skip_paths:
                            if skip_path in fn:
                                skip = True
                                break
                        if skip:
                            continue
                        v = self.db.get_fix_idx_vector(self.db.b2i(fix_idx)).reshape((512,))
                        images[count, :] = v
                        count += 1
                        self.db.put_idx(i, fix_idx)
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
                        if self.faces:
                            annotations = self.db.get_faces(fix_idx)
                            for face_idx, annotation in enumerate(annotations):
                                faces_array[faces_count, :] = annotation['embedding'][0].reshape((512,))
                                faces_count += 1
                                self.db.put_idx_face(faces_i, fix_idx, face_idx)
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
                    faiss.write_index(index, str(self.path_prefix / "images.index"))
                    if faces:
                        print(f"Saving faces index...")
                        faiss.write_index(faces_index, str(self.path_prefix / "faces.index"))

        print(f"Indexed {i} images and {faces_i} faces.")
        print(f"Done!")

    def run(self, base_paths):
        try:
            self.clip_paths(base_paths)
        except KeyboardInterrupt:
            print(f"Interrupted!")

        self.index_images()


if __name__ == '__main__':
    scanner = Scanner()
    scanner.run(sys.argv[1:])
