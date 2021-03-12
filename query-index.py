import time
import os
import os.path
import sys
import re
import numpy as np
import lmdb
import torch_device
import torch
import clip
import faiss
from PIL import Image, ExifTags
import cv2

import config
import database
from faces import annotate as annotate_faces

def go(j, go_dir, compensate):
    if go_dir == 0:
        return j + 1, compensate + 1
    return max(0, j + go_dir), max(0, compensate + go_dir)

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


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


class Search:

    def __init__(self, device=None):
        if device == None:
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "cpu"
        self.device = device

        self.model, transform = clip.load("ViT-B/32", device=device, jit=False)

        self.model.eval()

        database.open_db()
        config.open_db()

        self.index = faiss.read_index("images.index")
        self.index.nprobe = config.get_setting_int("probe", 64)

        self.faces_index = faiss.read_index("faces.index")
        self.faces_index.nprobe = config.get_setting_int("probe", 64)

        self.in_text = ""
        self.texts = None
        self.features = None
        self.show_faces = config.get_setting_bool("show_faces", False)
        self.show_prefix = config.get_setting_bool("show_prefix", True)
        self.face_threshold = config.get_setting_float("face_threshold", 0.60)
        self.clip_threshold = config.get_setting_float("clip_threshold", 0.19)
        self.k = config.get_setting_int("k", 50)
        self.offset = 0
        self.last_j = -1
        self.search_mode = -1
        self.max_res = None
        if config.get_setting_bool("max_res_set", False):
            self.max_res = (config.get_setting_int("max_res_x", 1280), config.get_setting_int("max_res_y", 720))
        self.align_window = config.get_setting_bool("align_window", False)
        self.results = None
        self.skip_same = config.get_setting_bool("skip_same", True)
        self.growth_limit = config.get_setting_int("growth_limit", 2**16)
        self.last_vector = None
        self.face_features = None
        self.file_filter = None
        self.file_filter_mode = True # Inverted?
        self.target_tag = None
        self.skip_perfect = config.get_setting_bool('skip_perfect', False)
        self.cluster_mode = False

    def run_cli(self):  # (CLI: command-line interface)
        print("A similar set of commands to the 't' commands exists with 'c'.\n"
              "These cluster based tags are completely separate from the regular ones.\n"
              "They require the clustering script to be run after building an index.\n"
              "Some commands behave differently, depending on whether 't' or 'c' has\n"
              "been least recently used.\n"
              "\n"
              "For help, type: h"
             )
        try:
            while self.in_text != 'q':
                # Handle commands
                prefix = "[h,q,l,i,if,t,T,t+,t-,t?,c!,ff,s,sp,r,a,n,ft,ct,p,k,gl] "
                if not self.show_prefix:
                    prefix = ""
                self.in_text = input(prefix + ">>> ").strip()
                iteration_done = self.do_command()
                if iteration_done:
                    continue

                self.do_search()
                self.do_display()
        except EOFError:
            print("Goodbye.")
        except KeyboardInterrupt:
            print("Interrupted.")

    def do_command(self):
        if self.in_text == 'q':
            return True
        elif self.in_text == 'h':
            print("Enter a search query and you will receive a list of best matching\n"
                  "images. The first number is the difference score, the second the\n"
                  "image ID followed by the filename.\n"
                  "\n"
                  "Press q to stop viewing image and space for the next image. Go back\n"
                  "by pressing a or backspace.\n"
                  "\n"
                  "For a nicer image viewing experience, try something like:\n"
                  ">>> a\n"
                  ">>> r 1600x900\n"
                  "\n"
                  "While tag searching, press + in the window to add the green\n"
                  "detection to the tag. Press - to remove yellow the yellow frame.\n"
                  "\n"
                  "A similar set of commands to the 't' commands exists with 'c'.\n"
                  "These cluster based tags are completely separate from the regular ones.\n"
                  "They require the clustering script to be run after building an index.\n"
                  "Some commands behave differently, depending on whether 't' or 'c' has\n"
                  "been least recently used.\n"
                  "\n"
                  "Just press enter for more results.\n"
                  "\n"
                  "Commands:\n"
                  "q\t\tQuit\n"
                  "l ID\t\tShow the image with the given ID and list faces\n"
                  "i ID\t\tFind images similar to ID\n"
                  "if ID F [S]\tFind images with faces similar to face F in image ID with optional query S\n"
                  "t TAG [S]\tFind images with faces tagged TAG and optional query S\n"
                  "T TAG [S]\tLike 't', but an average face embedding is added to the search\n"
                  "t+ TAG ID F\tAdd face F from image ID to tag TAG\n"
                  "t- TAG ID F\tRemove face F from image ID from tag TAG\n"
                  "t? TAG\t\tList which faces from which images belong to TAG\n"
                  "tl\t\tList tags exist\n"
                  "c/C/c+/c-/c?/cl\tLike the t commands, but affecting separate cluster tags instead\n"
                  "c! TAG ID F\tScrub all entries of the same cluster as face F from image ID from TAG\n"
                  "C! TAG ID F\tLike 'c!', but also permanently declusters all images from the cluster\n"
                  "cul [CID]\tList unnamed clusters with image number, or list images in unnamed cluster CID\n"
                  "fs\t\tToggle skipping full matches with 1.0 score\n"
                  "ff [RE]\t\tSet filename filter regular expression\n"
                  "ff!\t\tToggle filename filter inversion\n"
                  "s\t\tToggle display of on-image face annotations\n"
                  "sp\t\tToggle whether to show prompt prefix\n"
                  "r [RES]\t\tSet maximum resolution (e.g. 1280x720)\n"
                  "a\t\tToggle align window position\n"
                  "n NUM\t\tSet default number of results to NUM\n"
                  "ft THRES\tSet face similarity cutoff point in [0, 1] (default: 0.3)\n"
                  "ct THRES\tSet clip similarity cutoff point in [0, 1] for mixed search (default: 0.19)\n"
                  "p NUM\t\tSet number of subsets to probe (1-100, 32 default)\n"
                  "k\t\tSkip images with identical CLIP features\n"
                  "gl NUM\t\tSet maximum internal search result number (default: 65536)\n"
                  "h\t\tShow this help"
                 )
            return True
        elif self.in_text.startswith('gl '):
            try:
                limit = int(self.in_text[3:])
                if limit < 0:
                    raise Exception
                self.growth_limit = limit
                config.set_setting_float("growth_limit", self.growth_limit)
                print(f"Set search growth limit to {growth_limit}.")
            except:
                print("Invalid search growth limit.")
            return True
        elif self.in_text.startswith('ct '):
            try:
                threshold = float(self.in_text[3:])
                if threshold < 0.0 or threshold > 1.0:
                    raise Exception
                self.clip_threshold = threshold
                config.set_setting_float("clip_threshold", self.clip_threshold)
                print(f"Set CLIP similarity threshold to {clip_threshold}.")
            except:
                print("Invalid CLIP threshold value.")
            return True
        elif self.in_text.startswith('ft '):
            try:
                threshold = float(self.in_text[3:])
                if threshold < 0.0 or threshold > 1.0:
                    raise Exception
                self.face_threshold = threshold
                config.set_setting_float("face_threshold", self.face_threshold)
                print(f"Set face similarity threshold to {face_threshold}.")
            except:
                print("Invalid face threshold value.")
            return True
        elif self.in_text.startswith('p '):
            try:
                probe = int(self.in_text[2:])
                if probe <= 0 or probe > 100:
                    raise Exception
                self.index.nprobe = probe
                self.faces_index.nprobe = probe
                config.set_setting_int("probe", probe)
                print(f"Set to probe {probe} subsets.")
            except:
                print("Invalid probe value.")
            return True
        elif self.in_text == 'fs':
            self.skip_perfect = not self.skip_perfect
            config.set_setting_bool("skip_perfect", self.skip_perfect)
            if self.skip_perfect:
                print("Skipping perfect matches.")
            else:
                print("Not skipping perfect matches images.")
            return True
        elif self.in_text == 'k':
            self.skip_same = not self.skip_same
            config.set_setting_bool("skip_same", self.skip_same)
            if self.skip_same:
                print("Skipping images with the same CLIP features as previous.")
            else:
                print("Not skipping images.")
            return True
        elif self.in_text == 's':
            self.show_faces = not self.show_faces
            config.set_setting_bool("show_faces", self.show_faces)
            if self.show_faces:
                print("Showing face information.")
            else:
                print("Not showing face information.")
            return True
        elif self.in_text == 'sp':
            self.show_prefix = not self.show_prefix
            config.set_setting_bool("show_prefix", self.show_prefix)
            if self.show_prefix:
                print("Showing prompt prefix.")
            else:
                print("Not prompt prefix.")
            return True
        elif self.in_text == 'ff!':
            self.file_filter_mode = not self.file_filter_mode
            if self.file_filter_mode:
                print("Showing files with matching filenames.")
            else:
                print("Skipping files with matching filenames.")
            return True
        elif self.in_text == 'ff':
            self.file_filter = None
            print("Disabled filename filter.")
            return True
        elif self.in_text == 'a':
            self.align_window = not self.align_window
            config.set_setting_bool("align_window", self.align_window)
            if self.align_window:
                print("Aligning window position.")
            else:
                print("Not aligning window position.")
            return True
        elif self.in_text.startswith('ff '):
            self.file_filter = re.compile(self.in_text[3:])
            print("Set filename filter regular expression.")
            return True
        elif self.in_text.startswith('r '):
            res = self.in_text[2:]
            try:
                x, y = res.split('x')
                x = int(x)
                y = int(y)
                if x > 0 and y > 0:
                    self.max_res = (x, y)
                    config.set_setting_bool("max_res_set", True)
                    config.set_setting_int("max_res_x", x)
                    config.set_setting_int("max_res_y", y)
                    print(f"Set maximum resolution to {x}x{y}.")
                    return True
            except:
                pass
            self.max_res = None
            config.set_setting_bool("max_res_set", False)
            print("Unset maximum resolution.")
            return True
        elif self.in_text.startswith('n '):
            try:
                self.k = int(self.in_text[2:])
                if self.k < 1:
                    self.k = 50
                    print("Reset number of results to 50.")
                    return True
                print(f"Showing {self.k} results.")
            except:
                print("Error")
            return True
        elif self.in_text == 'tl' or self.in_text == 'cl':
            self.cluster_mode = self.in_text[0] == 'c'
            print("Existing tags:")
            print("#Faces\tTag")
            tags = sorted(config.list_tags(self.cluster_mode), key=lambda x: x[1])
            for tag in tags:
                num, name = tag
                print(f"{num}\t{name}")
            return True
        elif self.in_text == 'cul':
            self.cluster_mode = True
            print("Existing unnamed clusters:")
            print("ID\t#Faces")
            clusters = config.list_unnamed_clusters()
            for cluster in clusters:
                cluster_id, num = cluster
                print(f"{cluster_id}\t{num}")
            return True
        elif self.in_text.startswith('cul '):
            self.cluster_mode = True
            try:
                cluster_id = int(self.in_text[4:])
                self.results = config.get_unnamed_cluster_contents(cluster_id)
                if self.results is None or len(self.results) < 1:
                    print("Not found.")
                    return True
                self.offset = -1
                self.last_j = -1
                print(f"Showing unnamed cluster {cluster_id}:")
                self.search_mode = -2
                self.last_vector = None
            except:
                print("Unnamed cluster not found.")
        elif self.in_text.startswith('t+ ') or self.in_text.startswith('c+ '):
            self.cluster_mode = self.in_text[0] == 'c'
            try:
                parts = self.in_text[3:].split(" ")
                tag = parts[0]
                image_id = int(parts[1])
                face_id = int(parts[2])
                if not config.add_tag(tag, image_id, face_id, self.cluster_mode):
                    raise Exception
                print(f"Added face {face_id} from image {image_id} to tag {tag}.")
            except:
                print("Adding to tag failed.")
            return True
        elif self.in_text.startswith('t- ') or self.in_text.startswith('c- '):
            self.cluster_mode = self.in_text[0] == 'c'
            try:
                parts = self.in_text[3:].split(" ")
                tag = parts[0]
                image_id = int(parts[1])
                face_id = int(parts[2])
                if not config.del_tag(tag, image_id, face_id, self.cluster_mode):
                    raise Exception
                print(f"Removed face {face_id} from image {image_id} from tag {tag}.")
            except:
                print("Removing from tag failed.")
            return True
        elif self.in_text.startswith('c! ') or self.in_text.startswith('C! '):
            self.cluster_mode = True
            try:
                prevent_recluster = self.in_text[0] == 'C'
                parts = self.in_text[3:].split(" ")
                tag = parts[0]
                image_id = int(parts[1])
                face_id = int(parts[2])
                if not config.purge_cluster_tag(tag, image_id, face_id, prevent_recluster):
                    raise Exception
                print(f"Scrubbed the cluster of face {face_id} from image {image_id} from tag {tag}.")
            except:
                print("Scrubbing from tag failed.")
            return True
        elif self.in_text.startswith('t? ') or self.in_text.startswith('c? '):
            self.cluster_mode = self.in_text[0] == 'c'
            try:
                tag = self.in_text[3:]
                self.results = config.get_tag_contents(tag, self.cluster_mode)
                if self.results is None or tag == "" or len(self.results) < 1:
                    print("Not found.")
                    return True
                self.offset = -1
                self.last_j = -1
                print(f"Showing tag {tag}:")
                print(f"Image\tFace")
                for result, score in self.results:
                    print(f"{result[0]}\t{result[1]}")
                self.search_mode = -2
                self.target_tag = tag
                self.last_vector = None
            except:
                print("Error")
                return True
        elif self.in_text.startswith('l '):
            try:
                image_id = int(self.in_text[2:])
                self.offset = -1
                self.last_j = -1
                try:
                    filename = database.get_fix_idx_filename(image_id)
                    features = database.get_fix_idx_vector(image_id)
                    annotations = database.get_faces(database.i2b(image_id))
                    print(f"Showing {filename}:")
                    print(f"Image\tFace\tTag\tBounding box")
                    for i, annotation in enumerate(annotations):
                        tag = config.get_face_tag(annotation, self.face_threshold, self.cluster_mode)
                        print(f"{image_id}\t{i}\t{tag}\t{annotation['bbox']}")
                except:
                    print("Not found.")
                    return True
                self.results = [(image_id, 1.0)] * self.k
                self.search_mode = -2
                self.target_tag = None
                self.last_vector = None
            except:
                print("Error")
                return True
        elif self.in_text.startswith('t ') or self.in_text.startswith('T ') or self.in_text.startswith('c ') or self.in_text.startswith('C '):
            self.cluster_mode = self.in_text[0] == 'c'
            try:
                self.search_mode = 1
                self.last_vector = None
                parts = self.in_text[2:].split(" ", 2)
                tag = parts[0]
                self.target_tag = tag
                self.offset = -1
                self.last_j = -1

                self.features = config.get_tag_embeddings(tag, self.cluster_mode)
                if self.in_text.startswith('T '):
                    average = self.features.mean(0, keepdims=True)
                    average = normalize(average)
                    self.features = np.append(self.features, average, axis=0)
                if self.features is None:
                    print("Not found.")
                    self.search_mode = -1
                    self.last_vector = None
                    return True
                if len(parts) > 1:
                    self.face_features = self.features
                    self.search_mode = 2
                    self.texts = clip.tokenize([parts[1]]).to(self.device)
                    self.features = normalize(self.model.encode_text(self.texts).detach().cpu().numpy().astype('float32'))
                print(f"Similar faces to {tag}:")
            except:
                print("Error")
                return True
        elif self.in_text.startswith('if '):
            try:
                self.search_mode = 1
                self.target_tag = None
                self.last_vector = None
                parts = self.in_text[3:].split(" ", 3)
                image_id = int(parts[0])
                face_id = int(parts[1])
                self.offset = -1
                self.last_j = -1

                filename = database.get_fix_idx_filename(image_id)
                annotations = database.get_faces(database.i2b(image_id))
                self.features = annotations[face_id]['embedding']
                if len(parts) > 2:
                    self.face_features = self.features
                    self.search_mode = 2
                    self.texts = clip.tokenize([parts[2]]).to(self.device)
                    self.features = normalize(self.model.encode_text(self.texts).detach().cpu().numpy().astype('float32'))
                print(f"Similar faces to {face_id} in {image_id}:")
            except:
                print("Not found.")
                self.search_mode = -1
                self.last_vector = None
                return True
        elif self.in_text.startswith('i '):
            try:
                image_id = int(self.in_text[2:])
                self.offset = -1
                self.last_j = -1
                filename = database.get_fix_idx_filename(image_id)
                self.features = database.get_fix_idx_vector(image_id)
                print(f"Similar to {filename}:")
            except:
                print("Not found.")
                return True
            self.search_mode = 0
            self.target_tag = None
            self.last_vector = None
        elif self.in_text == '':
            self.offset = self.last_j
            if self.features is None and self.search_mode > 0:
                return True
        else:
            self.offset = -1
            self.last_j = -1
            self.texts = clip.tokenize([self.in_text]).to(self.device)
            self.features = normalize(self.model.encode_text(self.texts).detach().cpu().numpy().astype('float32'))
            self.search_mode = 0
            self.target_tag = None
            self.last_vector = None

    def do_search(self):
        if self.search_mode == 0:
            # Search CLIP features
            search_start = time.perf_counter()
            last_results_num = -1
            I = [[]]
            extra = 0
            valid_results = 0
            while valid_results < self.k + self.offset + 2 and len(I[0]) > last_results_num:
                last_results_num = len(I[0])
                D, I = self.index.search(self.features, self.k + self.offset + 2 + extra)
                self.results = merge_faiss_results(D, I, database.get_idx)
                if self.file_filter is None:
                    break
                if extra == 0:
                    extra = 64
                else:
                    extra *= 2
                if extra > self.growth_limit:
                    break
                valid_results = 0
                for result in self.results:
                    tfn = ""
                    if type(result[0]) is tuple:
                        tfn = database.get_fix_idx_filename(result[0][0])
                    else:
                        tfn = database.get_fix_idx_filename(result[0])
                    if (re.search(self.file_filter, tfn) is None) != self.file_filter_mode:
                        valid_results += 1
            search_time = time.perf_counter() - search_start
            print(f"Search time: {search_time:.4f}s")
        elif self.search_mode == 1:
            # Search face embedding
            search_start = time.perf_counter()
            D, I = self.faces_index.search(self.features, self.k + self.offset + 2)
            self.results = merge_faiss_results(D, I, database.get_idx_face)
            search_time = time.perf_counter() - search_start
            print(f"Search time: {search_time:.4f}s")
        elif self.search_mode == 2:
            # Search CLIP features containing face embedding
            search_start = time.perf_counter()
            _, D, I = self.faces_index.range_search(x=self.face_features, thresh=self.face_threshold)
            face_results = merge_faiss_results_1d_dict(D, I, database.get_idx_face)
            _, D, I = self.index.range_search(x=self.features, thresh=self.clip_threshold)
            np.set_printoptions(threshold=np.inf)
            self.results = merge_faiss_results_1d(D, I, database.get_idx, face_results)
            search_time = time.perf_counter() - search_start
            print(f"Search time: {search_time:.4f}s")

    def do_display(self):
        compensate = 0
        n_results = 0
        if self.results is not None:
            n_results = len(self.results)
        j = self.last_j + 1
        go_dir = 1
        tried_j = -1
        while j < n_results:
            if j < 0:
                j = 0
            if j - compensate >= self.offset + self.k:
                break
            j = min(max(j, 0), n_results - 1)
            result = self.results[j]
            if self.search_mode == 1 and result[1] < self.face_threshold:
                break
            fix_idx = None
            face_id = None
            output = ""
            self.last_j = j
            if type(result[0]) is tuple:
                face_id = result[0][1]
                fix_idx = result[0][0]
                tfn = database.get_fix_idx_filename(fix_idx)
                vector = database.get_fix_idx_vector(fix_idx)
                if self.last_vector is not None and np.array_equal(vector, self.last_vector) and self.search_mode != -2 and tried_j != j:
                    tried_j = j
                    j, compensate = go(j, go_dir, compensate)
                    continue
                self.last_vector = vector
                output = f"{result[1]:.4f} {fix_idx} {face_id} {tfn}"
            else:
                fix_idx = result[0]
                tfn = database.get_fix_idx_filename(fix_idx)
                vector = database.get_fix_idx_vector(fix_idx)
                if self.last_vector is not None and np.array_equal(vector, self.last_vector) and self.search_mode != -2:
                    j, compensate = go(j, go_dir, compensate)
                    continue
                self.last_vector = vector
                output = f"{result[1]:.4f} {fix_idx} {tfn}"
            if self.file_filter is not None:
                if (re.search(self.file_filter, tfn) is None) == self.file_filter_mode:
                    j, compensate = go(j, go_dir, compensate)
                    continue
            if self.skip_perfect and (result[1] > 0.999999 or config.has_tag(self.target_tag, fix_idx, face_id, self.cluster_mode)) and tried_j != j:
                    tried_j = j
                    j, compensate = go(j, go_dir, compensate)
                    continue
            tried_j = -1
            annotations = None
            if self.show_faces or self.target_tag is not None:
                annotations = database.get_faces(database.i2b(fix_idx))
                found_tag = False
                for a_i, annotation in enumerate(annotations):
                    annotation['tag'] = config.get_face_tag(annotation, self.face_threshold, self.cluster_mode)
                    if face_id is not None and a_i == face_id and result[1] > 0.99999:
                        annotation['color'] = (0, 255, 255)
                    if self.target_tag is not None and (annotation['tag'] == self.target_tag or (self.cluster_mode and annotation['tag'] == "")):
                        found_tag = True
                if self.target_tag is not None and not found_tag:
                    j, compensate = go(j, go_dir, compensate)
                    continue
            try:
                image = cv2.imread(tfn, cv2.IMREAD_COLOR)
                if image is None or image.shape[0] < 2:
                    j, compensate = go(j, go_dir, compensate)
                    continue
                h, w, _ = image.shape
                scale = 1.0
                if self.max_res is not None:
                    need_resize = False
                    if w > self.max_res[0]:
                        factor = float(self.max_res[0])/float(w)
                        w = self.max_res[0]
                        h *= factor
                        need_resize = True
                        scale *= factor
                    if h > self.max_res[1]:
                        factor = float(self.max_res[1])/float(h)
                        h = self.max_res[1]
                        w *= factor
                        need_resize = True
                        scale *= factor
                    if need_resize:
                        image = cv2.resize(image, (int(w + 0.5), int(h + 0.5)), interpolation=cv2.INTER_LANCZOS4)
                if self.show_faces:
                    pillow_image = Image.open(tfn)
                    exif_data = pillow_image._getexif()
                    exif_orientation = None
                    if exif_data is not None and orientation in exif_data:
                        exif_orientation = exif_data[orientation]
                    image = annotate_faces(annotations, image=image, scale=scale, face_id=face_id, orientation=exif_orientation, skip_landmarks=True)
                cv2.imshow('Image', image)
                if self.align_window:
                    cv2.moveWindow('Image', 0, 0)
                key = ""
                do_break = False
                print(output)
                while key != ord(" "):
                    key = cv2.waitKey(0) & 0xff
                    if key == ord('q'):
                        do_break = True
                        break
                    elif key == ord(' '):
                        go_dir = 1
                        break
                    elif key == ord('a') or key == 8:
                        go_dir = -1
                        break
                    elif key == ord('+'):
                        go_dir = 1
                        if self.target_tag is not None:
                            result[1] = 1.0
                            config.add_tag(self.target_tag, fix_idx, face_id, self.cluster_mode)
                        break
                    elif key == ord('-'):
                        go_dir = 1
                        if self.target_tag is not None:
                            result[1] = self.face_threshold + 0.00001
                            config.del_tag(self.target_tag, fix_idx, face_id, self.cluster_mode)
                        break
                if do_break:
                    break
            except:
                j, compensate = go(j, go_dir, compensate)
                continue
            j = j + go_dir
        cv2.destroyAllWindows()


if __name__ == '__main__':

    # Might not exist, but let's try
    try:
        import readline
    except:
        pass

    search = Search()
    search.run_cli()
