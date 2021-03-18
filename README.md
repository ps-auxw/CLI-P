# CLI-P: Semantic commandline image search using CLIP

CLI-P is an easy, commandline interface driven way for the lazy
photographer to look for something in that big old image archive. I
quickly hacked it together, but it seems to be working alright.

It is a set of two python scripts to use
[CLIP](https://github.com/openai/CLIP) to search through a library of
image files from a simple commandline. It supports both search by text
embedding or image similarity, similar to
[same.energy](https://same.energy/).

CLI-P uses [LMDB](https://symas.com/lmdb/) as for storing filenames and
512D feature vectors and
[faiss](https://github.com/facebookresearch/faiss) to build a fast
similarity search index, so it should scale reasonably well to even high
numbers of images.

Face detection is being done with Retinaface, while face embeddings are
calculated with a pretrained ArcFace model.

Make sure you have virtualenv and swig installed.

To set things up, follow (or run) `setup.sh`. You may want to look at
the first few lines of `build-index.py` and adjust the variables there
to your liking. If you run it, you should get a message `All done.` at
the end. If you get an error about running out of pages for the LMDB,
rename `cli-p.conf.example` to `cli-p.conf` and customize it.

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

For a nicer image viewing experience, try:

    a
    r 1600x900

## Known issues

* Running `query-index.py` while a new index is being generated will
    give bad results.
* Running multiple instances of `build-index.py` in parallel may mess
    things up, but running a single instance of it afterwards should
    fix them back up.

## Testing

There's a (currently very incomplete, but) emerging test suite
which you can call via: `python -m unittest discover tests`

That is, invoke the Python `unittest` module's `discover` sub-command
on the `tests` directory.

## Credits

By ps-auxw. OO/GUI work by canvon.
