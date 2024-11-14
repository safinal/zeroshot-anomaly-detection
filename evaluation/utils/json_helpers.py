# -----------------------------------------------------------------------------
#  Do Not Alter This File!
# -----------------------------------------------------------------------------
#  The following code is part of the logic used for loading and evaluating your
#  output scores. Please DO NOT modify this section, as upon your submission,
#  the whole evaluation logic will be overwritten by the original code.
# -----------------------------------------------------------------------------

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {
                "__ndarray__": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        else:
            return super(NumpyEncoder, self).default(obj)


def dict_to_json(dct, filename):
    """Save a dictionary to a JSON file"""
    with open(filename, "w") as f:
        json.dump(dct, f, cls=NumpyEncoder)


def json_to_dict(filename):
    """Load a JSON file and convert it back to a dictionary of NumPy arrays"""
    with open(filename, "r") as f:
        dct = json.load(f)

    for k, v in dct.items():
        if isinstance(v, dict) and "__ndarray__" in v:
            dct[k] = np.array(v["__ndarray__"], dtype=v["dtype"]).reshape(v["shape"])

    return dct
