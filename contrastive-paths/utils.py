"""
Utility functions for contrastive path learning.
This module contains helper functions for working with tonnetz coordinates and paths.
"""

from imports import *


# TODO(neeraja): make this a method of EFGenus
def ratio_to_coord(ratio, ef):
    """ Find the coordinates of a ratio within the EF Genus
    
    Args:
        ratio: Fraction or float representing the frequency ratio
        ef: EFGenus object containing prime numbers and powers
        
    Returns:
        numpy.ndarray: Coordinates of the ratio in the EF genus space
    """
    ratio = Fraction(ratio)  # ensure it's a Fraction
    # prime factorization of the numerator
    numerator = [0,]*len(ef.primes)
    denominator = [0,]*len(ef.primes)
    rn = ratio.numerator
    rd = ratio.denominator
    for ii, pp in enumerate(ef.primes):
        while rn % pp == 0:
            numerator[ii] += 1
            rn = int(rn / pp)
        while rd % pp == 0:
            denominator[ii] += 1
            rd = int(rd / pp)
    # convert to coordinates
    ratio_coords = np.array(numerator) - np.array(denominator)
    return ratio_coords


def encode_hf(coordinate, saptak_mark, tn):
    """
    Encode a coordinate with saptak mark into harmono-frequency space.
    
    Args:
        coordinate: Tuple of coordinates in the tonnetz
        saptak_mark: String representing saptak (octave) marking
        tn: Tonnetz object
        
    Returns:
        Tuple: (*coordinate, freq) where freq is the frequency ratio
        
    TODO: replace the input with a Shruti object
    """
    if saptak_mark not in SAPTAK_MARKS:
        raise ValueError(f"Invalid saptak mark: {saptak_mark}. Must be one of {SAPTAK_MARKS}.")
    freq = float(tn.coord_to_ratio(coordinate))
    return (*coordinate, freq)


def create_harmono_frequency_space_paths(samooha, ground_truth, tn):
    """
    Create good and bad paths in harmono-frequency space for a given samooha.
    
    Args:
        samooha: list[SSwar] - sequence of swara with saptak marks
        ground_truth: list[tuple[int]] - ground truth coordinate tuples 
        tn: Tonnetz object
        
    Returns:
        dict: Dictionary with 'good_paths' and 'bad_paths' keys containing
              lists of paths in harmono-frequency space
    """
    bad_paths = []
    good_paths = []
    
    ground_truth_coordinates = [ratio_to_coord(b, tn.ef if hasattr(tn, 'ef') else EFGenus(tn.primes, tn.powers)) for b in ground_truth]
    ground_truth_coordinates = [tuple(row) for row in ground_truth_coordinates]
    
    # take a cartesian product of all options
    all_options = []
    all_saptak_marks = [str(ss)[:-1] for ss in samooha]
    for ss in samooha:
        all_options.append(tn.get_swar_options(ss.swar.name))
    all_paths = itertools.product(*all_options)
    
    for path in all_paths:
        # if all elements of the path are in tn.node_coordinates[ground_truth]:
        if all([cc in ground_truth_coordinates for cc in path]):
            # this is a good path
            good_paths.append([
                encode_hf(coordinate, saptak_mark, tn)
                for coordinate, saptak_mark in zip(path, all_saptak_marks)
            ])
        else:
            # this is a bad path
            bad_paths.append([
                encode_hf(coordinate, saptak_mark, tn)
                for coordinate, saptak_mark in zip(path, all_saptak_marks)
            ])
    
    return {
        "good_paths": good_paths,
        "bad_paths": bad_paths,
    }


def add_to_dataset(r_phrases, r_ground_truth, tn, all_samples_train, all_samples_val, add_to_split=None):
    """
    Add phrase data to training or validation datasets.
    
    Args:
        r_phrases: list of strings - raag phrases 
        r_ground_truth: list of Fraction objects - ground truth ratios
        tn: Tonnetz object
        all_samples_train: list to append training samples to
        all_samples_val: list to append validation samples to  
        add_to_split: str - "val" to add to validation, otherwise adds to training
        
    Returns:
        None (modifies the input lists in place)
    """
    for phrase in r_phrases:
        samples = create_harmono_frequency_space_paths(
            [SSwar.from_string(s) for s in phrase.split(" ")], 
            r_ground_truth, 
            tn
        )
        if add_to_split == "val":
            all_samples_val.append(samples)
        else:
            all_samples_train.append(samples)