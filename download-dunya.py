token = "185d809a541e65d254bc577774a3fce1749bbf50"

import codecs
import json, os, sys
import pickle
import csv
import time
import datetime
import collections

import numpy as np

import compmusic

from compmusic import dunya as dn
from compmusic.dunya import hindustani as hi
from compmusic.dunya import carnatic as ca
from compmusic.dunya import docserver as ds
from compmusic import musicbrainz

dn.set_token(token)

# Features list
features_dunya_all = [
    {
        "type": "pitch",
        "subtype": "pitch",
        "extension": ".pitch",
        "version": "noguessunv",
    },
    {"type": "ctonic", "subtype": "tonic", "extension": ".tonic", "version": "0.3"},
    {"type": "sama-manual", "subtype": None, "extension": ".sama", "version": None},
    {
        "type": "sections-manual",
        "subtype": None,
        "extension": ".sections",
        "version": None,
    },
    {"type": "tempo-manual", "subtype": None, "extension": ".tempo", "version": None},
    {"type": "pitch-vocal", "subtype": None, "extension": ".mpitch", "version": None},
    {
        "type": "mphrases-manual",
        "subtype": None,
        "extension": ".mphrases",
        "version": None,
    },
    {
        "type": "sections-manual-p",
        "subtype": None,
        "extension": ".sections_p",
        "version": None,
    },
    {"type": "bpm-manual", "subtype": None, "extension": ".bpm", "version": None},
]


def getStatsDunyaCorpus():
    """
    Compute and save statistics for the Hindustani and Carnatic collections.

    Outputs:
        A Pickle to 'stats_{collection}_cc.pkl' of the MusicBrainz IDs that appear in the collection
        A text file to 'stats_{collection}_cc.txt' showing summary counts of items that appear in the collection
    """

    carnatic_stats = get_stats_carnatic(DUNYA_COLLECTIONS["carnatic"])
    output_file = "stats_carnatic_cc.pkl"
    output_file_pretty = "stats_carnatic_cc.txt"
    save_stats(carnatic_stats, output_file, output_file_pretty)

    hindustani_stats = get_stats_hindustani(DUNYA_COLLECTIONS["hindustani"])
    output_file = "stats_hindustani_cc.pkl"
    output_file_pretty = "stats_hindustani_cc.txt"
    save_stats(hindustani_stats, output_file, output_file_pretty)


def get_stats_hindustani(dunya_collections=None):
    """Get information about hindustani recordings and return a summary of attributes.
    For the following attributes:
        release
        works
        raags
        taals
        forms
        layas
        album artists
        artists (musicians)
    generate a list of identifiers for these attributes (mbid or uuid [raag, taal, laya] or name [form])
    present in the collection

    Args:
        dunya_collections: a list of MusicBrainz/Dunya Collection IDs to restrict the Dunya API to
    """

    hi.set_collections(dunya_collections)
    recordings = hi.get_recordings()

    stats = collections.defaultdict(list)
    for r in recordings:
        mbid = r["mbid"]

        try:
            rec_info = hi.get_recording(mbid)

            stats["release"].append([r["mbid"] for r in rec_info.get("release", [])])
            stats["works"].append([w["mbid"] for w in rec_info.get("works", [])])
            stats["raags"].append([r["uuid"] for r in rec_info.get("raags", [])])
            stats["taals"].append([t["uuid"] for t in rec_info.get("taals", [])])
            stats["forms"].append([f["name"] for f in rec_info.get("forms", [])])
            stats["layas"].append([l["uuid"] for l in rec_info.get("layas", [])])
            stats["album_artists"].append(
                [a["mbid"] for a in rec_info.get("album_artists", [])]
            )
            stats["artists"].append(
                [a["artist"]["mbid"] for a in rec_info.get("artists", [])]
            )
            stats["length"].append(rec_info.get("length"))
        except:
            failure += 1
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("Failed to fetch info for recording %s" % mbid)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Filter empty lists from the stats
    for k, vals in stats.items():
        stats[k] = [v for v in vals if v]

    return stats


def get_stats_carnatic(dunya_collections=None):
    """Get information about carnatic recordings and return a summary of attributes.
    For the following attributes:
        concert
        work
        raaga
        taala
        form
        album artists
        artists (musicians)
    generate a list of identifiers for these attributes (mbid or uuid [raaga, taala] or name [form])
    present in the collection

    Args:
        dunya_collections: a list of MusicBrainz/Dunya Collection IDs to restrict the Dunya API to
    """
    ca.set_collections(dunya_collections)
    recordings = ca.get_recordings()

    stats = collections.defaultdict(list)
    for r in recordings:
        mbid = r["mbid"]

        try:
            rec_info = ca.get_recording(mbid)

            stats["concert"].append([c["mbid"] for c in rec_info.get("concert", [])])
            stats["work"].append([w["mbid"] for w in rec_info.get("work", [])])
            stats["raaga"].append([r["uuid"] for r in rec_info.get("raaga", [])])
            stats["taala"].append([t["uuid"] for t in rec_info.get("taala", [])])
            stats["form"].append([f["name"] for f in rec_info.get("form", [])])
            stats["album_artists"].append(
                [a["mbid"] for a in rec_info.get("album_artists", [])]
            )
            stats["artists"].append(
                [a["artist"]["mbid"] for a in rec_info.get("artists", [])]
            )
            stats["length"].append(rec_info.get("length"))
        except dn.HTTPError:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("Failed to fetch info for recording %s" % mbid)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Filter empty lists from the stats
    for k, vals in stats.items():
        stats[k] = [v for v in vals if v]

    return stats


def save_stats(stats, stats_file, summary_file):
    """Write statistics to file
    Args:
        stats (dict): the statistics to write
        stats_file (str): file path to write the statistics summary
        summary_file (str): file path to write a readable statistics summary
    """

    output_stats = {}
    for k, v in stats.items():
        if k == "length":
            output_stats[k] = {"total_length": np.sum(v), "total_recs": len(v)}
        else:
            output_stats[k] = {
                "total_unique": len(np.unique(sum(v, []))),
                "unique_elems": np.unique(sum(v, [])).tolist(),
                "total_rels": len(sum(v, [])),
                "total_recs": len(v),
            }
    pickle.dump(output_stats, codecs.open(stats_file, "wb"))

    with codecs.open(summary_file, "w") as fp:
        for key1, val in output_stats.items():
            fp.write("------------ %s ------------\n" % str(key1))
            if key1 == "length":
                for key2, val2 in val.items():
                    fp.write("%s\t%f\n" % (str(key2), float(val2) / (1000.0 * 3600.0)))
            else:
                for key2, val2 in val.items():
                    if key2 == "unique_elems":
                        fp.write("%s\t%d\n" % (str(key2), len(val2)))
                    else:
                        fp.write("%s\t%d\n" % (str(key2), val2))
            fp.write("\n")


def saveSections(content, output_file):
    """
    This function saves the content(section annotations) into a file in a structured manner
    Annotations are already stored nicely but due to differences in the delimiters of Hindustani and Carnatic
    we needed this function

    Args:
        content (str): data read from dunya api
        output_file (str): file path for output file
    Outputs:
        Saves statistics to a text file
    """

    # detecting delimiter automatically
    snf = csv.Sniffer()
    delimiter = snf.sniff(content).delimiter
    rows = [k.split(delimiter) for k in content.split("\n") if k != ""]
    csv.writer(output_file, rows, delimiter="\t")


def download_files_for_collection(
    collection_name, collection_ids, features, startFiles=0, endFiles=5
):
    """Download all files of a collection
    Args:
        collection (dict): dictionary containig name and id of the collection
        features (list of dicts): feature types
        startFiles, endFiles (int, int): the indexes of files to download
    Returns:
        A dictionary counting how many files for each feature was unable to be downloaded
    Outputs:
        Saves mp3 and annotation files of the collection
    """
    dataDir = collection_name
    os.makedirs(dataDir, exist_ok=True)

    if collection_name == "hindustani":
        tradition = hi
    elif collection_name == "carnatic":
        tradition = ca

    tradition.set_collections(collection_ids)
    recs = tradition.get_recordings()

    endFiles = startFiles + min(endFiles - startFiles, len(recs))

    print("Number of files in collection {}: {}".format(collection_name, len(recs)))
    print("...will download {} files".format(endFiles - startFiles))

    # Creating data structure for keeping list of missing files
    missingData = collections.Counter()
    for feature in features:
        missingData[feature["type"]] = 0

    # Downloading data
    for i, recording in enumerate(recs[startFiles:endFiles], 1):
        mbid = recording["mbid"]
        print("{}/{}: {}".format(i, len(recs), mbid))
        mp3_filename = tradition.download_mp3(mbid, dataDir)
        # mp3_filename = "Ajoy Chakrabarty - Raag Jog.mp3"
        json_file = mp3_filename.replace(".mp3", ".json")
        with open(os.path.join(dataDir, json_file), "w") as outfile:
            json.dump(tradition.get_recording(mbid), outfile)

        print(mp3_filename)

        for feature in features:
            print(f"\t {feature}")
            try:
                content = ds.file_for_document(
                    mbid,
                    feature["type"],
                    feature["subtype"],
                    version=feature["version"],
                )

                out_file = os.path.join(
                    dataDir,
                    mp3_filename.replace(".mp3", ".{}.txt".format(feature["type"])),
                )
                if feature["type"] == "pitch":
                    content = json.loads(content.decode())
                    content = np.array(content)
                    np.savetxt(out_file, content, fmt="%.7f", delimiter="\t")
                # elif feature['type'] == 'sections-manual' or feature['type'] == 'sections-manual-p':
                #    saveSections(content.decode(), out_file)
                else:
                    with open(out_file, "w") as fp:
                        fp.write(content.decode())
            except dn.HTTPError:
                # print('Does not have ',feature['type'],' content for :',mbid)
                missingData[feature["type"]] += 1

    print("Collection download finished.")
    print("----------------------------------------------------------")
    return dict(missingData)


print("Starting process: {}".format(datetime.datetime.now()))

print("Downloading files ... ")
missingDatas = {}
missingData = download_files_for_collection(
    "data-dunya-hindustani",
    ["6adc54c6-6605-4e57-8230-b85f1de5be2b"],
    features_dunya_all,
    0,
    108,
)
missingDatas["hindustani"] = missingData
print("...Done")

pickle.dump(missingDatas, codecs.open("missingData.pkl", "wb"))
print("Missing data list stored in missingData.pkl")

print("Finished! {}".format(datetime.datetime.now()))
