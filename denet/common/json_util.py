import os
import gzip
import numpy
import base64
import json
import io

def numpy_to_json(python_object):
    if isinstance(python_object, numpy.ndarray):
        strio = io.BytesIO()
        numpy.save(strio, python_object)

        return {'__class__': 'numpy.ndarray',
                '__value__': base64.b64encode(strio.getvalue()).decode() }

    raise TypeError(type(python_object))

def numpy_from_json(json_object):
    if '__class__' in json_object:
        if json_object['__class__'] == 'numpy.ndarray':
            bio = io.BytesIO(base64.b64decode(json_object['__value__']))
            return numpy.load(bio)

    return json_object

def json_from_gz(fname):
    with gzip.open(fname, "rt") as f:
        json_obj=json.load(f, object_hook=numpy_from_json)
    return json_obj

def json_to_gz(fname, json_obj, compresslevel=9):
    with gzip.open(fname, "wt", compresslevel=compresslevel) as f:
        json.dump(json_obj, f, indent=2, default=numpy_to_json)

def json_from_file(fname):
    if os.path.splitext(fname)[1] == ".gz":
        return json_from_gz(fname)
    else:
        with open(fname, "rt") as f:
            json_obj=json.load(f, object_hook=numpy_from_json)
        return json_obj

def json_to_file(fname, json_obj):
    if os.path.splitext(fname)[1] == ".gz":
        return json_to_gz(fname, json_obj)
    else:
        with open(fname, "wt") as f:
            json.dump(json_obj, f, indent=2, default=numpy_to_json)
