import inspect
from IPython.core.magics.code import extract_symbols

import inspect, sys


def new_getfile(object, _old_getfile=inspect.getfile):
    if not inspect.isclass(object):
        return _old_getfile(object)

    # Lookup by parent module (as in current inspect)
    if hasattr(object, "__module__"):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, "__file__"):
            return object_.__file__

    # If parent module is __main__, lookup by methods (NEW)
    for name, member in inspect.getmembers(object):
        if (
            inspect.isfunction(member)
            and object.__qualname__ + "." + member.__name__ == member.__qualname__
        ):
            return inspect.getfile(member)
    else:
        raise TypeError("Source for {!r} not found".format(object))


inspect.getfile = new_getfile


def get_cell_code(obj):
    return "".join(inspect.linecache.getlines(new_getfile(obj)))


def get_class_code(obj):
    cell_code = get_cell_code(obj)
    return extract_symbols(cell_code, obj.__name__)[0][0]


def get_members(obj):
    members = {}
    for i in inspect.getmembers(obj):
        if not i[0].startswith("_") and not inspect.ismethod(i[1]):
            members[i[0]] = i[1]
    return members
