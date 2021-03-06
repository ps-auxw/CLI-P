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

import database
from faces import annotate as annotate_faces

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 0.000000001: 
       return v
    return v / norm

def merge_faiss_results(D, I, lookup):
    results = {}
    for i, indexes in enumerate(I):
        for j, index in enumerate(indexes):
            if index < 0:
                continue
            if index not in results or D[i][j] > results[index][1]:
                results[index] = [lookup(index), D[i][j]]
    return sorted(results.values(), key=lambda x: x[1], reverse=True)

def merge_faiss_results_1d(D, I, lookup, check_list):
    results = {}
    for i, index in enumerate(I):
        if index < 0:
            continue
        fix_idx = lookup(index)
        if index not in results or D[i] > results[index][0]:
            if str(fix_idx) in check_list:
                results[index] = [fix_idx, D[i]]
    return sorted(results.values(), key=lambda x: x[1], reverse=True)

def merge_faiss_results_1d_dict(D, I, lookup):
    results = {}
    for i, index in enumerate(I):
        if index < 0:
            continue
        fix_idx = lookup(index)
        if index not in results or D[i] > results[str(fix_idx[0])]:
            results[str(fix_idx[0])] = D[i]
    return results

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, transform = clip.load("ViT-B/32", device=device, jit=False)

model.eval()

database.open_db()

index = faiss.read_index("images.index")
index.nprobe = 32

faces_index = faiss.read_index("faces.index")
faces_index.nprobe = 32

in_text = ""
texts = None
features = None
show_faces = False
face_threshold = 0.3
clip_threshold = 0.19
k = 50
offset = 0
last_j = 0
search_mode = -1
max_res = None
align_window = False
results = None
skip_same = True
last_vector = None
face_features = None
try:
    while in_text != 'q':
        # Handle commands
        in_text = input("[h,q,l,i,if,s,r,a,c,ft,ct,p,k] >>> ").strip()
        if in_text == 'q':
            break
        elif in_text == 'h':
            print("Enter a search query and you will receive a list of best matching\nimages. The first number is the difference score, the second the\nimage ID followed by the filename.\n\nPress q to stop viewing image and space for the next image.\n\nJust press enter for more results.\n\nCommands:\nq\tQuit\nl ID\tShow the image with the given ID and list faces\ni ID\tFind images similar to ID\nif ID F\tFind images with faces similar to face F in image ID\ns\tToggle display of on-image face annotations\nr [RES]\tSet maximum resolution (e.g. 1280x720)\na\tToggle align window position\nc NUM\tSet default number of results to NUM\nft THRES\tSet face similarity cutoff point in [0, 1]\nct THRES\tSet clip similarity cutoff point in [0, 1] for mixed search\np NUM\tSet number of subsets to probe (1-100, 32 default)\nk\tSkip images with identical CLIP features\nh\tShow this help")
            continue
        elif in_text.startswith('ct '):
            threshold = float(in_text[3:])
            if threshold >= 0.0 and threshold <= 1.0:
                clip_threshold = threshold
                print(f"Set CLIP similarity threshold to {clip_threshold}.")
                continue
            print("Invalid CLIP threshold value.")
            continue
        elif in_text.startswith('ft '):
            threshold = float(in_text[3:])
            if threshold >= 0.0 and threshold <= 1.0:
                face_threshold = threshold
                print(f"Set face similarity threshold to {face_threshold}.")
                continue
            print("Invalid face threshold value.")
            continue
        elif in_text.startswith('p '):
            probe = int(in_text[2:])
            if probe > 0 and probe < 101:
                index.nprobe = probe
                faces_index.nprobe = probe
                print(f"Set to probe {probe} subsets.")
                continue
            print("Invalid probe value.")
            continue
        elif in_text == 'k':
            skip_same = not skip_same
            if skip_same:
                print("Skipping images with the same CLIP features as previous.")
            else:
                print("Not skipping images.")
            continue
        elif in_text == 's':
            show_faces = not show_faces
            if show_faces:
                print("Showing face information.")
            else:
                print("Not showing face information.")
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
        elif in_text.startswith('l '):
            image_id = int(in_text[2:])
            offset = -1
            last_j = 0
            try:
                filename = database.get_fix_idx_filename(image_id)
                features = database.get_fix_idx_vector(image_id)
                annotations = database.get_faces(database.i2b(image_id))
                print(f"Showing {filename}:")
                print(f"Image\tFace\tBounding box")
                for i, annotation in enumerate(annotations):
                    print(f"{image_id}\t{i}\t{annotation['bbox']}")
            except:
                print("Not found.")
                continue
            results = [(image_id, 1.0)] * k
            search_mode = -1
        elif in_text.startswith('if '):
            try:
                search_mode = 1
                parts = in_text[3:].split(" ", 3)
                image_id = int(parts[0])
                face_id = int(parts[1])
                offset = -1
                last_j = 0

                filename = database.get_fix_idx_filename(image_id)
                annotations = database.get_faces(database.i2b(image_id))
                features = annotations[face_id]['embedding']
                if len(parts) > 2:
                    face_features = features
                    search_mode = 2
                    texts = clip.tokenize([parts[2]]).to(device)
                    features = normalize(model.encode_text(texts).detach().cpu().numpy().astype('float32'))
                print(f"Similar faces to {face_id} in {image_id}:")
            except:
                print("Not found.")
                search_mode = -1
                continue
        elif in_text.startswith('i '):
            image_id = int(in_text[2:])
            offset = -1
            last_j = 0
            try:
                filename = database.get_fix_idx_filename(image_id)
                features = database.get_fix_idx_vector(image_id)
                print(f"Similar to {filename}:")
            except:
                print("Not found.")
                continue
            search_mode = 0
        elif in_text == '':
            offset = last_j
            if features is None and search_mode > 0:
                continue
        else:
            offset = -1
            last_j = 0
            texts = clip.tokenize([in_text]).to(device)
            features = normalize(model.encode_text(texts).detach().cpu().numpy().astype('float32'))
            search_mode = 0

        # Do search
        if search_mode == 0:
            # Search CLIP features
            search_start = time.perf_counter()
            D, I = index.search(features, k + offset + 2)
            results = merge_faiss_results(D, I, database.get_idx)
            search_time = time.perf_counter() - search_start
            print(f"Search time: {search_time:.4f}s")
        elif search_mode == 1:
            # Search face embedding
            search_start = time.perf_counter()
            D, I = faces_index.search(features, k + offset + 2)
            results = merge_faiss_results(D, I, database.get_idx_face)
            search_time = time.perf_counter() - search_start
            print(f"Search time: {search_time:.4f}s")
        elif search_mode == 2:
            # Search CLIP features containing face embedding
            search_start = time.perf_counter()
            _, D, I = faces_index.range_search(x=face_features, thresh=face_threshold)
            face_results = merge_faiss_results_1d_dict(D, I, database.get_idx_face)
            _, D, I = index.range_search(x=features, thresh=clip_threshold)
            results = merge_faiss_results_1d(D, I, database.get_idx, face_results)
            search_time = time.perf_counter() - search_start
            print(f"Search time: {search_time:.4f}s")

        # Do display
        for j, result in enumerate(results):
            if j <= offset:
                continue
            if j >= offset + k:
                break
            if search_mode == 1 and result[1] < face_threshold:
                break
            face_id = None
            output = ""
            last_j = j
            if type(result[0]) is tuple:
                face_id = result[0][1]
                result[0] = result[0][0]
                tfn = database.get_fix_idx_filename(result[0])
                vector = database.get_fix_idx_vector(result[0])
                if last_vector is not None and np.array_equal(vector, last_vector):
                    continue
                last_vector = vector
                output = f"{result[1]:.4f} {result[0]} {face_id} {tfn}"
            else:
                tfn = database.get_fix_idx_filename(result[0])
                vector = database.get_fix_idx_vector(result[0])
                if last_vector is not None and np.array_equal(vector, last_vector):
                    continue
                last_vector = vector
                output = f"{result[1]:.4f} {result[0]} {tfn}"
            try:
                image = cv2.imread(tfn, cv2.IMREAD_COLOR)
                if image is None or image.shape[0] < 2:
                    continue
                h, w, _ = image.shape
                scale = 1.0
                if max_res is not None:
                    need_resize = False
                    if w > max_res[0]:
                        factor = float(max_res[0])/float(w)
                        w = max_res[0]
                        h *= factor
                        need_resize = True
                        scale *= factor
                    if h > max_res[1]:
                        factor = float(max_res[1])/float(h)
                        h = max_res[1]
                        w *= factor
                        need_resize = True
                        scale *= factor
                    if need_resize:
                        image = cv2.resize(image, (int(w + 0.5), int(h + 0.5)), interpolation=cv2.INTER_LANCZOS4)
                if show_faces:
                    annotations = database.get_faces(database.i2b(result[0]))
                    image = annotate_faces(annotations, image=image, scale=scale, face_id=face_id)
                cv2.imshow('Image', image)
                if align_window:
                    cv2.moveWindow('Image', 0, 0)
                key = ""
                do_break = False
                print(output)
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
