import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return round(float(obj), 2)  
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def resize_by_shortside(size, shortside=256):
    w, h = size
    if w > h:
        long = w / h * shortside
        return [int(long), shortside]
    else:
        long = h / w * shortside
        return [shortside, int(long)]