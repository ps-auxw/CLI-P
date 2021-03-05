import os
import os.path
import sys
import numpy as np
import lmdb
import torch
import clip
import faiss
from PIL import Image

# Maximum size of (sparse) LMDB file
map_size = 1024*1024*1024*20

# If you run out of RAM building a #Images * 512 floats numpy array, set this to build sequential 20k * 512 floats tables instead (might disimprove search quality)
split_table = False 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device, jit=False)

model.eval()

env = lmdb.open('vectors.lmdb', map_size=map_size, max_dbs=4)
fn_db = env.open_db("fn_db".encode())
skip_db = env.open_db("skip_db".encode())

try:
    with torch.no_grad():
        for base_path in sys.argv[1:]:
            print(f"CLIPing {base_path}...")
            for fn in os.listdir(base_path):
                tfn = base_path + fn
                ext = os.path.splitext(fn)
                if len(ext) < 2 or not ext[1].lower() in ['.jpg', '.jpeg', '.png']:
                    continue
                skipped = False
                with env.begin(db=skip_db) as skip_txn:
                    try:
                        if skip_txn.get(tfn.encode()) is not None:
                            continue
                    except:
                        continue
                with env.begin(db=fn_db, write=True) as txn:
                    if txn.get(tfn.encode()) is not None:
                        continue
                    image = None
                    try:
                        image = Image.open(tfn)
                        image = transform(image).unsqueeze(0).to(device)
                        image_features = model.encode_image(image)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        txn.put(tfn.encode(), image_features.detach().cpu().numpy().astype('float32').tobytes())
                        print(".", end="", flush=True)
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt()
                    except:
                        print("#", end="", flush=True)
                        skipped = True
                        continue
                if skipped:
                    with env.begin(db=skip_db, write=True) as skip_txn:
                        skip_txn.put(tfn.encode(), "1".encode())
            print(flush=True)
except KeyboardInterrupt:
    print(f"Interrupted!")

idx_db = env.open_db("idx_db".encode())

with env.begin(db=fn_db) as txn:
    n = txn.stat()['entries']
    nd = n
    count = 0
    need_training = True
    if split_table and nd > 20000:
        nd = 20000
    cursor = txn.cursor()
    if cursor.first():
        images = np.zeros((nd, 512))
        i = 0
        print(f"Preparing index for {n} entries...")
        quantizer = faiss.IndexFlatIP(512)
        index = faiss.IndexIVFFlat(quantizer, 512, 100, faiss.METRIC_INNER_PRODUCT)
        print(f"Generating {images.shape} matrix...")
        for tfn, vector in cursor:
            v = np.frombuffer(vector, dtype=np.float32)
            v = v.reshape((512,))
            images[count, :] = v
            with env.begin(db=idx_db, write=True) as idx_txn:
                idx_txn.put(f"{i}".encode(), tfn, dupdata=False, overwrite=True)
            i += 1
            count += 1
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
        if count > 0:
            images = images[0:count].astype('float32')
            if need_training:
                print(f"Training index {images.shape}...")
                index.train(images)
            print(f"Adding to index...")
            index.add(images)
        print(f"Saving index...")
        faiss.write_index(index, "images.index")

print(f"Done!")

env.close()
