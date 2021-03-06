# CLI-P: Semantic commandline image search using CLIP

CLI-P is an easy, commandline interface driven way for the lazy
photographer to look for something in that big old image archive. I
quickly hacked it together, but it seems to be working alright.

It is a set of two python scripts to use [CLIP](https://github.com/openai/CLIP)
to search through a library of image files from a simple commandline. It
supports both search by text embedding or image similarity, similar to
[same.energy](https://same.energy/).

CLI-P uses [LMDB](https://symas.com/lmdb/) as for storing filenames and 512D
feature vectors and [faiss](https://github.com/facebookresearch/faiss) to build
a fast similarity search index, so it should scale reasonably well to even high
numbers of images.

To set things up, follow (or run) `setup.sh`. You may want to look at
the first few lines of `build-index.py` and adjust the two variable
there to your liking.

If you are running Debian buster, install cmake 3.16 from
`buster-backports` first.

## Indexing images in a folder

Make sure you have activated the virtual environment in your current
shell using:

    . env/bin/activate

Run `python build-index.py` with a list of folders as arguments. It will
not recurse into subfolder. Only files ending with `.jpg`, `.jpeg` or
`.png` will be indexed, but you could add other extensions in the code.

## Searching for images

Make sure you have activated the virtual environment in your current
shell using:

    . env/bin/activate

Run `python query-index.py` and you will get a prompt where you can
enter search terms and a few commands. Enter `h` for a short help.

## Known issues

* Running `query-index.py` while a new index is being generated will
    give bad results.
* Running multiple instances of `build-index.py` in parallel may mess
    things up, but running a single instance of it afterwards should
    fix them back up.
