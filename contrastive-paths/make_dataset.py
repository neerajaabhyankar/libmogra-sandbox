from imports import *
from utils import ratio_to_coord, encode_hf, create_harmono_frequency_space_paths, add_to_dataset


def quick_test():
    primes = [3, 5]
    ef = EFGenus(primes=primes, powers=[4, 1])
    tn = Tonnetz(ef)

    bmpls = [1, Fraction(9,8), Fraction(6, 5), Fraction(4,3), Fraction(3,2), Fraction(27,16), Fraction(9,5)]
    b = ",n S g m P".split(" ")
    b = [SSwar.from_string(s) for s in b]
    samples = create_harmono_frequency_space_paths(b, bmpls, tn)

    assert len(samples["good_paths"]) == 1
    assert len(samples["bad_paths"]) == 47


RAAG_DATASET = {
    "Bheempalasi": {
        "ground_truth": [1, Fraction(9,8), Fraction(6, 5), Fraction(4,3), Fraction(3,2), Fraction(27,16), Fraction(9,5)],
        "phrases": [
            ",n S g m P",
            "n D P",
            "m P g m",
            ",n S g R S",
            "m g R S",
        ],
    },
    "Darbari": {
        "ground_truth": [Fraction(128,81), Fraction(32,27), Fraction(10, 9), Fraction(16,9), Fraction(4,3), Fraction(1), Fraction(3,2), Fraction(40, 27)],
        "phrases": [
            "S R g g",
            "m P d d n P",
            "n n P m P g",
            "m P g g m R S",
            ",n S R S ,d ,d ,n S",
            "`S d d n P",
            "R R S ,n S",
            ",d ,n R",
        ],
    },
    "Puriya Dhanashree": {
        "ground_truth": [Fraction(1), Fraction(3,2), Fraction(5,4), Fraction(15,8), Fraction(45,32), Fraction(135, 128), Fraction(405, 256)],
        "phrases": [
            ",N r G M P",
            "P d M P",
            "M G M r G",
            "M d N `S",
            "N `r N d P",
            "M d M G",
            "N `r `G `r `S",
            "N d N `r N d P",
        ],
    },
    "Bhairav": {
        "ground_truth": [Fraction(1), Fraction(3,2), Fraction(5,4), Fraction(15,8), Fraction(4,3), Fraction(16,15), Fraction(8,5)],
        "phrases": [
            "G m r r S",
            "`S N d d P",
            "G m d d P",
            ",N S r r S",
            "G m P G m",
            "G m N d",
            "d N `S",
        ],
    },
}

SPLIT = {
    "train": ["Bheempalasi", "Darbari", "Puriya Dhanashree"],
    "val": ["Bhairav"],
}

primes = [3, 5]
ef = EFGenus(primes=primes, powers=[4, 1])
tn = Tonnetz(ef)
all_samples_train = []
all_samples_val = []

def add_to_dataset_helper(r_phrases, r_ground_truth, add_to_split=None):
    """Helper function that uses the global tn object"""
    global tn, all_samples_train, all_samples_val
    add_to_dataset(r_phrases, r_ground_truth, tn, all_samples_train, all_samples_val, add_to_split)

for split, raag_names in SPLIT.items():
    for raag_name in raag_names:
        add_to_dataset_helper(RAAG_DATASET[raag_name]["phrases"], RAAG_DATASET[raag_name]["ground_truth"], add_to_split=split)

# assert that each phrase has at least 1 good path
assert all([len(samples["good_paths"]) > 0 for samples in all_samples_train]), "Not all phrases have good paths"
