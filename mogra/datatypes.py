import math
from fractions import Fraction
import bisect
from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union


PRIMES = [3, 5, 7, 11]
SAPTAK_MARKS = OrderedDict({",,": -2, ",": -1, "": 0, "`": 1, "``": 2})
SWAR_BOUNDARIES = [
    1.026749,
    1.088889,
    1.155093,
    1.225,
    1.299479,
    1.378125,
    1.452655,
    1.540123,
    1.633333,
    1.732639,
    1.8375,
    1.949219,
]


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

    @staticmethod
    def from_string(string):
        string = string.strip()
        if len(string) > 1:
            return SSwar(string[:-1], string[-1])
        else:
            return SSwar("", string)

    def __str__(self):
        return list(SAPTAK_MARKS)[self.saptak.value + 2] + self.swar.name

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        return self.swar == other.swar


def normalize_frequency(ff: Union[float, Fraction]):
    """Bring any relative frequency within the primary octave"""
    while ff < 1:
        ff *= 2
    while ff >= 2:
        ff /= 2
    return ff


def ratio_to_swar(ff: Union[float, Fraction]) -> Swar:
    """
    Per the swar boundaries, returns the coarse-grained Swar symbol that ff may map to
    """
    si = bisect.bisect_left(SWAR_BOUNDARIES, float(ff))
    return Swar(si % 12).name


def ratio_to_swarval(ff: Union[float, Fraction]):
    """
    With the octave represented as [0, 12), and equal-temperament-tuned notes as integers,
    returns the real-valued "note value" given a ratio between [1, 2)
    """
    return math.log2(float(ff)) * 12


class Shruti:
    def __init__(
        self,
        num_denom: Optional[Tuple[int, int]] = None,
        powers: Optional[Tuple] = None,
    ) -> None:
        self.ratio = Fraction(1)
        if num_denom is not None:
            self.ratio = Fraction(num_denom[0], num_denom[1])
        elif powers is not None:
            for ii, pp in enumerate(powers):
                self.ratio *= PRIMES[ii] ** pp

        self.frequency = normalize_frequency(self.ratio)
        self.swar = ratio_to_swar(self.frequency)


class Samooha:
    def __init__(self, string) -> None:
        self.list = []
        ii = 0
        while ii < len(string):
            if (ii < len(string) - 1) and string[ii + 1] in SAPTAK_MARKS:
                self.list.append(SSwar(string[ii + 1], string[ii]))
                ii += 1
            else:
                try:
                    self.list.append(SSwar("", string[ii]))
                except Exception as e:
                    print("skipping char", string[ii], "; ", e)
            ii += 1


# @dataclass
# class Raag:
#     name: str
#     alt_names: List[str]
#     aaroha: List[Swar]
#     avaroha: List[Swar]
