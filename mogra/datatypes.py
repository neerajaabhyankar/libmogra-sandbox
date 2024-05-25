from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum
from typing import List, Dict


""" Types """


class Swar(Enum):
    S = 0
    r = 1
    R = 2
    g = 3
    G = 4
    m = 5  # shuddha!
    M = 6  # teevra!
    P = 7
    d = 8
    D = 9
    n = 10
    N = 11


SAPTAK_MARKS = OrderedDict({",,": -2, ",": -1, "": 0, "`": 1, "``": 2})


class Saptak(Enum):
    ati_mandra = -2
    mandra = -1
    madhya = 0
    taara = 1
    ati_taara = 2


class SSwar(object):
    def __init__(self, saptak_mark="", swar_name="S"):
        self.saptak = Saptak(SAPTAK_MARKS[saptak_mark])
        try:
            self.swar = Swar[swar_name]
        except:
            print(f"WARNING: NO SWAR {swar_name}")
            if swar_name in "ps":
                print(f"trying {swar_name.upper()}")
                self.swar = Swar[swar_name.upper()]

    def __str__(self):
        return list(SAPTAK_MARKS)[self.saptak.value + 2] + self.swar.name

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        return self.swar == other.swar


# @dataclass
# class Raag:
#     name: str
#     alt_names: List[str]
#     aaroha: List[Swar]
#     avaroha: List[Swar]
