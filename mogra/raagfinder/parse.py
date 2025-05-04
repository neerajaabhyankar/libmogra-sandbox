import os, sys
import pickle
from rich import print, table
import rapidfuzz
from .. import datatypes


""" setup """


def index_by_set(raag_db):
    # TODO: instead of an exact match, allow for a subset/one-away match
    # TODO: use a different data structure for this
    raag_db_by_set = {}
    for raag_name, raag_entry in raag_db.items():
        swar_set = raag_entry["aaroha"] + raag_entry["avaroha"]
        swar_set = [datatypes.SSwar.from_string(sw).swar for sw in swar_set]
        swar_set = [sw.name for sw in sorted(swar_set, key=lambda x: x.value)]
        swar_set = list(dict.fromkeys(swar_set))
        if tuple(swar_set) in raag_db_by_set:
            raag_db_by_set[tuple(swar_set)].append(raag_name)
        else:
            raag_db_by_set.update({tuple(swar_set): [raag_name]})

    return raag_db_by_set


def read_pickle():
    raag_db = pickle.load(
        open(os.path.join(os.path.dirname(__file__), "raags.pkl"), "rb")
    )
    return raag_db, index_by_set(raag_db)


RAAG_DB, RAAG_DB_BY_SWAR = read_pickle()


""" functions """


def best_match(raag_name):
    best_match = rapidfuzz.process.extractOne(raag_name, RAAG_DB.keys())
    return best_match[0]


def print_table(raag_entry):
    # table
    rich_table = table.Table()
    # columns
    rich_table.add_column("Attribute", justify="right", style="cyan", no_wrap=True)
    rich_table.add_column("Value", style="magenta")
    # rows
    for key, value in raag_entry.items():
        if key == "mukhyanga":
            value = "\n".join(
                "-- " + ", ".join(map(str, sublist)) for sublist in value
            )[:-4]
        elif isinstance(value, list):
            value = ", ".join(map(str, value))
        rich_table.add_row(key, str(value))

    # print
    print(rich_table)
