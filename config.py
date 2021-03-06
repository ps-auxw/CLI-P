import lmdb
import numpy as np
import struct
from numpack import *

# LMDB environment
env = None

# Information about face tags
tags_db = None

# Settings
settings_db = None

# uint64
def open_db(map_size=1024*1024*1024):
    global env, tags_db, settings_db
    env = lmdb.open('config.lmdb', map_size=map_size, max_dbs=5)

    tags_db = env.open_db(b'tags_db')
    settings_db = env.open_db(b'settings_db')

open_db()

def set_setting(name, value, conv):
    with env.begin(db=settings_db, write=True) as txn:
        txn.put(name.encode(), conv(value))

def get_setting(name, default, conv):
    try:
        with env.begin(db=settings_db) as txn:
            return conv(txn.get(name.encode()))
    except:
        return default

def set_setting_int(name, value):
    set_setting(name, value, i2b)

def get_setting_int(name, default):
    return get_setting(name, default, b2i)

def set_setting_bool(name, value):
    if value:
        set_setting(name, 1, i2b)
    else:
        set_setting(name, 0, i2b)

def get_setting_bool(name, default):
    if default:
        default = 1
    else:
        default = 0
    return get_setting(name, default, b2i) == 1

def set_setting_float(name, value):
    set_setting(name, value, f2b)

def get_setting_float(name, default):
    return get_setting(name, default, b2f)
