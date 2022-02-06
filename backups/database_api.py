from pathlib import Path
from uuid import uuid4
from tinydb import TinyDB, Query
from datetime import datetime
from pprint import pprint
from copy import deepcopy
from shutil import rmtree

instance = Query()
db = TinyDB(Path(__file__).resolve().parent / "db.json")
trainings = db.table("trainings")
models = db.table("models")
datasets = db.table("datasets")


def _unnest_dict(dict):
    keys = list(dict.keys())
    for key in keys:
        splited_keys = key.split(".")
        if len(splited_keys) > 1:
            dict[splited_keys[0]] = _unnest_dict(
                {".".join(splited_keys[1:]): dict.pop(key)}
            )
    return dict


def _update_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, dict):
            tmp = _update_dict(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = orig_dict.get(key, []) + val
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def _get_table(db_name):
    if db_name == "trainings":
        return trainings
    elif db_name == "models":
        return models
    elif db_name == "datasets":
        return datasets
    raise ValueError(f"No table called {db_name}")


def create(obj, db="trainings"):
    table = _get_table(db)
    return table.insert(obj)


def exists(id, db="training"):
    return retrieve(id, db=db) != None


def retrieve(id, db="trainings"):
    table = _get_table(db)
    return table.get(doc_id=int(id))


def update(id, query, db="trainings"):
    table = _get_table(db)
    assert isinstance(query, dict)
    obj = retrieve(id, db=db)
    obj = _update_dict(obj, _unnest_dict(query))
    table.update(
        obj, doc_ids=[int(x) for x in id] if isinstance(id, list) else [int(id)]
    )


def delete(id, db="trainings"):
    table = _get_table(db)
    ids = [int(x) for x in id] if isinstance(id, list) else [int(id)]
    table.remove(doc_ids=ids)

    if db == "datasets":
        for id in ids:
            path = Path(__file__).resolve().parent / "datasets" / str(id)
            rmtree(path)

    if db == "models":
        for id in ids:
            path = Path(__file__).resolve().parent / "datasets" / str(id)
            rmtree(path)


if __name__ == "__main__":
    print(db)
    id = create()
    training = retrieve(id)
    print(training.doc_id)
    pprint(training)
    update(id, {"checkpoint.first": 3})
    print(retrieve(id))

    id = create_model("hvrh")

    id = create_dataset("hvrh")
