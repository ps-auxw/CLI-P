import time
import os
import os.path
import sys
import numpy as np
import lmdb
import torch
import clip
import faiss
from PIL import Image
import cv2

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 0.000000001: 
       return v
    return v / norm

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, transform = clip.load("ViT-B/32", device=device, jit=False)

model.eval()

env = lmdb.open('vectors.lmdb', map_size=1024*1024*1024*20, max_dbs=4)
idx_db = env.open_db("idx_db".encode())
fn_db = env.open_db("fn_db".encode())

index = faiss.read_index("images.index")
index.nprobe = 32

in_text = ""
texts = None
features = None
k = 50
offset = 0
last_j = 0
max_res = None
align_window = False
try:
    while in_text != 'q':
        in_text = input("[h,q,i,r,a,c,p] >>> ").strip()
        if in_text == 'q':
            break
        elif in_text == 'h':
            print("Enter a search query and you will receive a list of best matching\nimages. The first number is the difference score, the second the\nimage ID followed by the filename.\n\nPress q to stop viewing image and space for the next image.\n\nJust press enter for more results.\n\nCommands:\nq\tQuit\ni ID\tFind images similar to ID\nr [RES]\tSet maximum resolution (e.g. 1280x720)\na\tToggle align window position\nc NUM\tSet default number of results to NUM\np NUM\tSet number of subsets to probe (1-100, 32 default)\nh\tShow this help")
            continue
        elif in_text.startswith('p '):
            probe = int(in_text[2:])
            if probe > 0 and probe < 101:
                index.nprobe = probe
                print(f"Set to probe {probe} subsets.")
                continue
            print("Invalid probe value.")
            continue
        elif in_text == 'a':
            align_window = not align_window
            if align_window:
                print("Aligning window position.")
            else:
                print("Not aligning window position.")
            continue
        elif in_text.startswith('r '):
            res = in_text[2:]
            try:
                x, y = res.split('x')
                x = int(x)
                y = int(y)
                if x > 0 and y > 0:
                    max_res = (x, y)
                    print(f"Set maximum resolution to {x}x{y}.")
                    continue
            except:
                pass
            max_res = None
            print("Unset maximum resolution.")
            continue
        elif in_text.startswith('c '):
            k = int(in_text[2:])
            if k < 1:
                k = 50
                print("Reset number of results to 50.")
                continue
            print(f"Showing {k} results.")
            continue
        elif in_text.startswith('i '):
            image_id = int(in_text[2:])
            offset = 0
            last_j = 0
            try:
                key = f"{image_id}".encode()
                with env.begin(db=idx_db) as txn:
                    key = txn.get(key)
                with env.begin(db=fn_db) as txn:
                    features = np.frombuffer(txn.get(key), dtype=np.float32).reshape((1,512))
                print(f"Similar to {key.decode()}:")
            except:
                print("Not found.")
                continue
        elif in_text == '':
            offset = last_j
            if texts is None:
                continue
        else:
            offset = 0
            last_j = 0
            texts = clip.tokenize([in_text]).to(device)
            features = normalize(model.encode_text(texts).detach().cpu().numpy().astype('float32'))

        search_start = time.perf_counter()
        D, I = index.search(features, k + offset + 1)
        search_time = time.perf_counter() - search_start
        print(f"Search time: {search_time:.4f}s")
        for j, i in enumerate(I[0]):
            if j <= offset:
                continue
            with env.begin(db=idx_db) as txn:
                tfn = txn.get(f"{i}".encode()).decode()
                print(f"{D[0][j]:.4f} {i} {tfn}")
                try:
                    last_j = j
                    image = cv2.imread(tfn, cv2.IMREAD_COLOR)
                    if image is None or image.shape[0] < 2:
                        continue
                    h, w, _ = image.shape
                    if max_res is not None:
                        need_resize = False
                        if w > max_res[0]:
                            factor = float(max_res[0])/float(w)
                            w = max_res[0]
                            h *= factor
                            need_resize = True
                        if h > max_res[1]:
                            factor = float(max_res[1])/float(h)
                            h = max_res[1]
                            w *= factor
                            need_resize = True
                        if need_resize:
                            image = cv2.resize(image, (int(w + 0.5), int(h + 0.5)), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imshow('Image', image)
                    if align_window:
                        cv2.moveWindow('Image', 0, 0)
                    key = ""
                    do_break = False
                    while key != ord(" "):
                        key = cv2.waitKey(0) & 0xff
                        if key == ord('q'):
                            do_break = True
                            break
                    if do_break:
                        break
                except:
                    continue
        cv2.destroyAllWindows()
except EOFError:
    print("Interrupted.")
except KeyboardInterrupt:
    print("Interrupted.")

sys.exit(0)
