import os
import hashlib
import math
import sys
from Frame_process.utils import get_value

"""
 The following modules are from : https://github.com/Orange-OpenSource/AIVC
 They are licenced under BSD 3-Clause "New"
 
 The licences on which our framework depends are to be found in the folder named "licence_dependency"
"""

def write_md5sum(param):
    """
    Write the md5sum of a given file in a binary file.
    This is used by the decoder to check wether their reconstruction is
    identical to the one at the encoder-side.
    """

    DEFAULT_PARAM = {
        # File on which we want to compute the md5sum
        'in_file': None,
        # File on which we want to save the md5sum
        'out_file': None,
    }

    in_file = get_value('in_file', param, DEFAULT_PARAM)
    out_file = get_value('out_file', param, DEFAULT_PARAM)

    md5_checksum = compute_md5sum({'in_file': in_file})
    with open(out_file, 'w') as fout:
        fout.write(md5_checksum)

def read_md5sum(param):
    """
    Return the md5sum written in a file
    """

    DEFAULT_PARAM = {
        # File in which we want to read the md5sum
        'in_file': None,
    }

    in_file = get_value('in_file', param, DEFAULT_PARAM)

    with open(in_file, 'r') as fin:
        md5sum = fin.read()
    
    return md5sum

def compute_md5sum(param):
    """
    Return the md5sum of a file a bytes
    """

    DEFAULT_PARAM = {
        # File on which we want to compute the md5sum
        'in_file': ''
    }

    in_file = get_value('in_file', param, DEFAULT_PARAM)

    # From stackoverflow
    # https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
    hash_md5 = hashlib.md5()
    with open(in_file, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()  # or hexdigest
