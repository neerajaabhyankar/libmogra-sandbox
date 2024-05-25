from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum
from typing import List, Dict
import pandas as pd
import re
from mogra.datatypes import SSwar


TR_TAARA_SAPTAK_MARKS = ["'", "`", "’"]
TR_PRAHAR_SECTIONS = ["day", "night"]
TR_PRAHAR_SUBECTIONS = ["1st", "2nd", "3rd", "4th"]


def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])


class TanarangParsedRaag:
    def __init__(self, df: pd.DataFrame, name="", verbose=True) -> None:
        self.df = df
        self.verbose = verbose
        self.name = name
        print(f"\n == parsing info of {self.name} ==")
        try:
            self._set_aaroha_avaroha()
        except:
            print("WARNING: UNABLE TO SET AAROHA/AVAROHA")
            if not hasattr(self, "aaroha"):
                self.aaroha = []
            if not hasattr(self, "avaroha"):
                self.avaroha = []
        try:
            self._set_mukhyanga()
        except:
            print("WARNING: UNABLE TO SET MUKHYANGA")
        try:
            self._set_nyas()
        except:
            print("WARNING: UNABLE TO SET NYAS")
        try:
            self._set_vaadi_samvaadi()
        except:
            print("WARNING: UNABLE TO SET VAADI/SAMVAADI")
        try:
            self._set_thaat_prahar()
        except:
            print("WARNING: UNABLE TO SET THAAT/PRAHAR")

    @staticmethod
    def string_to_swars(string):
        swars = []
        # TEMP: ignore the part after or!
        if "or" in string:
            string = string.split("or")[0]
        # TEMP: treat kan swars as regular swars
        string = string.replace("(", " ")
        string = string.replace(")", " ")
        # Add spaces between notes so they split correctly
        string = string.replace(",", " ,")
        for mm in TR_TAARA_SAPTAK_MARKS:
            string = string.replace(mm, mm + " ")
        string = re.sub(r"(?<=\w)(?=\w)", " ", string)
        for ss in string.split():
            ss = ss.strip("; ")
            if not re.search(r"\w", ss):
                continue
            if "," in ss:
                swars.append(SSwar(",", ss.strip(",")))
            elif any([ch in ss for ch in TR_TAARA_SAPTAK_MARKS]):
                swars.append(SSwar("`", ss.strip("".join(TR_TAARA_SAPTAK_MARKS))))
            else:
                swars.append(SSwar("", ss))
        return swars

    def _set_aaroha_avaroha(self):
        info = str(
            self.df.loc[self.df["info_type"].str.contains("roh")]["info"].values[0]
        )
        info_ar, info_av = info.split("-")
        self.aaroha: List[SSwar] = self.string_to_swars(info_ar)
        self.avaroha: List[SSwar] = self.string_to_swars(info_av)
        if self.verbose:
            print(f"aaroha {self.aaroha}")
            print(f"avaroha {self.avaroha}")

    def _set_mukhyanga(self):
        info = str(self.df.loc[self.df["info_type"] == "Mukhya-Ang"]["info"].values[0])
        phrases = info.split(";")
        self.mukhyanga: List[List[SSwar]] = [
            self.string_to_swars(phrase) for phrase in phrases
        ]
        if self.verbose:
            print(f"mukhyanga {self.mukhyanga}")

    def _set_nyas(self):
        info = str(
            self.df.loc[self.df["info_type"] == "Vishranti Sthan"]["info"].values[0]
        )
        try:
            info_ar, info_av = info.split("-")
        except:
            info_ar = info_av = info
        self.aarohi_nyas: List[SSwar] = self.string_to_swars(info_ar)
        self.avarohi_nyas: List[SSwar] = self.string_to_swars(info_av)
        if self.verbose:
            print(f"aarohi_nyas {self.aarohi_nyas}")
            print(f"avarohi_nyas {self.avarohi_nyas}")

    def _set_vaadi_samvaadi(self):
        info = str(
            self.df.loc[self.df["info_type"] == "Vadi/Samvadi"]["info"].values[0]
        )
        info_vd, info_sd = info.split("/")
        try:
            self.vaadi = SSwar("", info_vd[0])
            assert self.vaadi in self.aaroha + self.avaroha
        except:
            try:
                self.vaadi = SSwar("", info_vd[0].swapcase())
                assert self.vaadi in self.aaroha + self.avaroha
            except:
                print("WARNING: UNABLE TO SET VAADI = ", info_vd)
                self.vaadi = None
        try:
            self.samvaadi = SSwar("", info_sd[0])
            assert self.samvaadi in self.aaroha + self.avaroha
        except:
            try:
                self.samvaadi = SSwar("", info_sd[0].swapcase())
                assert self.samvaadi in self.aaroha + self.avaroha
            except:
                print("WARNING: UNABLE TO SET SAMVAADI = ", info_sd)
                self.samvaadi = None
        if self.verbose:
            print(f"vaadi {self.vaadi}")
            print(f"samvaadi {self.samvaadi}")

    def _set_thaat_prahar(self):
        self.thaat: str = str(
            self.df.loc[self.df["info_type"] == "Thaat"]["info"].values[0]
        )
        info = str(self.df.loc[self.df["info_type"] == "Time"]["info"].values[0])
        self.prahar: str = (
            TR_PRAHAR_SECTIONS[argmax([tt in info for tt in TR_PRAHAR_SECTIONS])]
            + " "
            + TR_PRAHAR_SUBECTIONS[argmax([tt in info for tt in TR_PRAHAR_SUBECTIONS])]
        )
        if self.verbose:
            print(f"thaat {self.thaat}")
            print(f"prahar {self.prahar}")

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["df"]
        del d["verbose"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
