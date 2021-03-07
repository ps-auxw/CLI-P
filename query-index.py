import time
import os
import os.path
import sys
import re
import numpy as np
import lmdb
import torch
import clip
import faiss
from PIL import Image, ExifTags
import cv2

import config
import database
from faces import annotate as annotate_faces

# Might not exist, but let's try
try:
    import readline
except:
    pass

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
            key = str(fix_idx)
            if key in check_list:
                results[index] = [(fix_idx, check_list[key][0]), D[i]]
    return sorted(results.values(), key=lambda x: x[1], reverse=True)

def merge_faiss_results_1d_dict(D, I, lookup):
    results = {}
    for i, index in enumerate(I):
        if index < 0:
            continue
        fix_idx = lookup(index)
        if index not in results or D[i] > results[str(fix_idx[0])][1]:
            results[str(fix_idx[0])] = [fix_idx[1], D[i]]
    return results

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, transform = clip.load("ViT-B/32", device=device, jit=False)

model.eval()

database.open_db()
config.open_db()

index = faiss.read_index("images.index")
index.nprobe = config.get_setting_int("probe", 64)

faces_index = faiss.read_index("faces.index")
faces_index.nprobe = config.get_setting_int("probe", 64)

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

in_text = ""
texts = None
features = None
show_faces = config.get_setting_bool("show_faces", False)
show_prefix = config.get_setting_bool("show_prefix", True)
face_threshold = config.get_setting_float("face_threshold", 0.3)
clip_threshold = config.get_setting_float("clip_threshold", 0.19)
k = config.get_setting_int("k", 50)
offset = 0
last_j = 0
search_mode = -1
max_res = None
if config.get_setting_bool("max_res_set", False):
    max_res = (config.get_setting_int("max_res_x", 1280), config.get_setting_int("max_res_y", 720))
align_window = config.get_setting_bool("align_window", False)
results = None
skip_same = config.get_setting_bool("skip_same", True)
growth_limit = config.get_setting_int("growth_limit", 2**16)
last_vector = None
face_features = None
file_filter = None
file_filter_mode = True # Inverted?
target_tag = None
try:
    while in_text != 'q':
        # Handle commands
        prefix = "[h,q,l,i,if,t,T,t+,t-,t?,ff,s,sp,r,a,c,ft,ct,p,k,gl] "
        if not show_prefix:
            prefix = ""
        in_text = input(prefix + ">>> ").strip()
        if in_text == 'q':
            break
        elif in_text == 'h':
            print("Enter a search query and you will receive a list of best matching\nimages. The first number is the difference score, the second the\nimage ID followed by the filename.\n\nPress q to stop viewing image and space for the next image.\n\nJust press enter for more results.\n\nCommands:\nq\tQuit\nl ID\tShow the image with the given ID and list faces\ni ID\tFind images similar to ID\nif ID F [S]\tFind images with faces similar to face F in image ID with optional query S\nt TAG [S]\tFind images with faces tagged TAG and optional query S\nT TAG [S]\tLike 't', but an average face embedding is added to the search\nt+ TAG ID F\tAdd face F from image ID to tag TAG\nt- TAG ID F\tRemove face F from image ID from tag TAG\nt? TAG\tList which faces from which images belong to TAG\nff [RE]\tSet filename filter regular expression\nff!\tToggle filename filter inversion\nToggle s\tToggle display of on-image face annotations\nsp\tToggle whether to show prompt prefix\nr [RES]\tSet maximum resolution (e.g. 1280x720)\na\tToggle align window position\nc NUM\tSet default number of results to NUM\nft THRES\tSet face similarity cutoff point in [0, 1] (default: 0.3)\nct THRES\tSet clip similarity cutoff point in [0, 1] for mixed search (default: 0.19)\np NUM\tSet number of subsets to probe (1-100, 32 default)\nk\tSkip images with identical CLIP features\ngl NUM\tSet maximum internal search result number (default: 65536)\nh\tShow this help")
            continue
        elif in_text.startswith('gl '):
            try:
                limit = int(in_text[3:])
                if limit < 0:
                    raise Exception
                growth_limit = limit
                config.set_setting_float("growth_limit", growth_limit)
                print(f"Set search growth limit to {growth_limit}.")
            except:
                print("Invalid search growth limit.")
            continue
        elif in_text.startswith('ct '):
            try:
                threshold = float(in_text[3:])
                if threshold < 0.0 or threshold > 1.0:
                    raise Exception
                clip_threshold = threshold
                config.set_setting_float("clip_threshold", clip_threshold)
                print(f"Set CLIP similarity threshold to {clip_threshold}.")
            finally:
                print("Invalid CLIP threshold value.")
            continue
        elif in_text.startswith('ft '):
            try:
                threshold = float(in_text[3:])
                if threshold < 0.0 or threshold > 1.0:
                    raise Exception
                face_threshold = threshold
                config.set_setting_float("face_threshold", face_threshold)
                print(f"Set face similarity threshold to {face_threshold}.")
            finally:
                print("Invalid face threshold value.")
            continue
        elif in_text.startswith('p '):
            try:
                probe = int(in_text[2:])
                if probe <= 0 or probe > 100:
                    raise Exception
                index.nprobe = probe
                faces_index.nprobe = probe
                config.set_setting_int("probe", probe)
                print(f"Set to probe {probe} subsets.")
            except:
                print("Invalid probe value.")
            continue
        elif in_text == 'k':
            skip_same = not skip_same
            config.set_setting_bool("skip_same", skip_same)
            if skip_same:
                print("Skipping images with the same CLIP features as previous.")
            else:
                print("Not skipping images.")
            continue
        elif in_text == 's':
            show_faces = not show_faces
            config.set_setting_bool("show_faces", show_faces)
            if show_faces:
                print("Showing face information.")
            else:
                print("Not showing face information.")
            continue
        elif in_text == 'sp':
            show_prefix = not show_prefix
            config.set_setting_bool("show_prefix", show_prefix)
            if show_prefix:
                print("Showing prompt prefix.")
            else:
                print("Not prompt prefix.")
            continue
        elif in_text == 'ff!':
            file_filter_mode = not file_filter_mode
            if file_filter_mode:
                print("Showing files with matching filenames.")
            else:
                print("Skipping files with matching filenames.")
            continue
        elif in_text == 'ff':
            file_filter = None
            print("Disabled filename filter.")
            continue
        elif in_text == 'a':
            align_window = not align_window
            config.set_setting_bool("align_window", align_window)
            if align_window:
                print("Aligning window position.")
            else:
                print("Not aligning window position.")
            continue
        elif in_text.startswith('ff '):
            file_filter = re.compile(in_text[3:])
            print("Set filename filter regular expression.")
            continue
        elif in_text.startswith('r '):
            res = in_text[2:]
            try:
                x, y = res.split('x')
                x = int(x)
                y = int(y)
                if x > 0 and y > 0:
                    max_res = (x, y)
                    config.set_setting_bool("max_res_set", True)
                    config.set_setting_int("max_res_x", x)
                    config.set_setting_int("max_res_y", y)
                    print(f"Set maximum resolution to {x}x{y}.")
                    continue
            except:
                pass
            max_res = None
            config.set_setting_bool("max_res_set", False)
            print("Unset maximum resolution.")
            continue
        elif in_text.startswith('c '):
            try:
                k = int(in_text[2:])
                if k < 1:
                    k = 50
                    print("Reset number of results to 50.")
                    continue
                print(f"Showing {k} results.")
            except:
                print("Error")
            continue
        elif in_text.startswith('t+ '):
            try:
                parts = in_text[3:].split(" ")
                tag = parts[0]
                image_id = int(parts[1])
                face_id = int(parts[2])
                if not config.add_tag(tag, image_id, face_id):
                    raise Exception
                print(f"Added face {face_id} from image {image_id} to tag {tag}.")
            except:
                print("Adding to tag failed.")
            continue
        elif in_text.startswith('t- '):
            try:
                parts = in_text[3:].split(" ")
                tag = parts[0]
                image_id = int(parts[1])
                face_id = int(parts[2])
                if not config.del_tag(tag, image_id, face_id):
                    raise Exception
                print(f"Removed face {face_id} from image {image_id} from tag {tag}.")
            except:
                print("Removing from tag failed.")
            continue
        elif in_text.startswith('t? '):
            try:
                tag = in_text[3:]
                results = config.get_tag_contents(tag)
                if results is None or tag == "" or len(results) < 1:
                    print("Not found.")
                    continue
                offset = -1
                last_j = 0
                print(f"Showing tag {tag}:")
                print(f"Image\tFace")
                for result, score in results:
                    print(f"{result[0]}\t{result[1]}")
                search_mode = -2
                target_tag = tag
                last_vector = None
            except:
                print("Error")
                continue
        elif in_text.startswith('l '):
            try:
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
                search_mode = -2
                target_tag = None
                last_vector = None
            except:
                print("Error")
                continue
        elif in_text.startswith('t ') or in_text.startswith('T '):
            try:
                search_mode = 1
                last_vector = None
                parts = in_text[2:].split(" ", 2)
                tag = parts[0]
                target_tag = tag
                offset = -1
                last_j = 0

                features = config.get_tag_embeddings(tag)
                if in_text.startswith('T '):
                    average = features.mean(0, keepdims=True)
                    average = normalize(average)
                    features = np.append(features, average, axis=0)
                if features is None:
                    print("Not found.")
                    search_mode = -1
                    last_vector = None
                    continue
                if len(parts) > 1:
                    face_features = features
                    search_mode = 2
                    texts = clip.tokenize([parts[1]]).to(device)
                    features = normalize(model.encode_text(texts).detach().cpu().numpy().astype('float32'))
                print(f"Similar faces to {tag}:")
            except:
                print("Error")
                continue
        elif in_text.startswith('if '):
            try:
                search_mode = 1
                target_tag = None
                last_vector = None
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
                last_vector = None
                continue
        elif in_text.startswith('i '):
            try:
                image_id = int(in_text[2:])
                offset = -1
                last_j = 0
                filename = database.get_fix_idx_filename(image_id)
                features = database.get_fix_idx_vector(image_id)
                print(f"Similar to {filename}:")
            except:
                print("Not found.")
                continue
            search_mode = 0
            target_tag = None
            last_vector = None
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
            target_tag = None
            last_vector = None

        # Do search
        if search_mode == 0:
            # Search CLIP features
            search_start = time.perf_counter()
            last_results_num = -1
            I = [[]]
            extra = 0
            valid_results = 0
            while valid_results < k + offset + 2 and len(I[0]) > last_results_num:
                last_results_num = len(I[0])
                D, I = index.search(features, k + offset + 2 + extra)
                results = merge_faiss_results(D, I, database.get_idx)
                if file_filter is None:
                    break
                if extra == 0:
                    extra = 64
                else:
                    extra *= 2
                if extra > growth_limit:
                    break
                valid_results = 0
                for result in results:
                    tfn = ""
                    if type(result[0]) is tuple:
                        tfn = database.get_fix_idx_filename(result[0][0])
                    else:
                        tfn = database.get_fix_idx_filename(result[0])
                    if (re.search(file_filter, tfn) is None) != file_filter_mode:
                        valid_results += 1
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
            np.set_printoptions(threshold=np.inf)
            results = merge_faiss_results_1d(D, I, database.get_idx, face_results)
            search_time = time.perf_counter() - search_start
            print(f"Search time: {search_time:.4f}s")

        # Do display
        compensate = 0
        for j, result in enumerate(results):
            if j - compensate <= offset:
                continue
            if j - compensate >= offset + k:
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
                if last_vector is not None and np.array_equal(vector, last_vector) and search_mode != -2:
                    compensate += 1
                    continue
                last_vector = vector
                output = f"{result[1]:.4f} {result[0]} {face_id} {tfn}"
            else:
                tfn = database.get_fix_idx_filename(result[0])
                vector = database.get_fix_idx_vector(result[0])
                if last_vector is not None and np.array_equal(vector, last_vector) and search_mode != -2:
                    compensate += 1
                    continue
                last_vector = vector
                output = f"{result[1]:.4f} {result[0]} {tfn}"
            if file_filter is not None:
                if (re.search(file_filter, tfn) is None) == file_filter_mode:
                    compensate += 1
                    continue
            annotations = None
            if show_faces or target_tag is not None:
                annotations = database.get_faces(database.i2b(result[0]))
                found_tag = False
                for a_i, annotation in enumerate(annotations):
                    annotation['tag'] = config.get_face_tag(annotation['embedding'], face_threshold)
                    if face_id is not None and a_i == face_id and result[1] > 0.99999:
                        annotation['color'] = (0, 255, 255)
                    if target_tag is not None and annotation['tag'] == target_tag:
                        found_tag = True
                if target_tag is not None and not found_tag:
                    compensate += 1
                    continue
            try:
                image = cv2.imread(tfn, cv2.IMREAD_COLOR)
                if image is None or image.shape[0] < 2:
                    compensate += 1
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
                    pillow_image = Image.open(tfn)
                    exif_data = pillow_image._getexif()
                    exif_orientation = None
                    if exif_data is not None and orientation in exif_data:
                        exif_orientation = exif_data[orientation]
                    image = annotate_faces(annotations, image=image, scale=scale, face_id=face_id, orientation=exif_orientation, skip_landmarks=True)
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
                compensate += 1
                continue
        cv2.destroyAllWindows()
except EOFError:
    print("Interrupted.")
except KeyboardInterrupt:
    print("Interrupted.")

sys.exit(0)
