import re
# helper class of amino acids
from enum import Enum


def remove_brackets(x): return re.sub(r'\[(.*?)\]', '', x).replace(" ", "")


def dummy_aligner(seq1, seq2):
    """ Aligns two sequences by adding spaces where no match and | if exact match.
    """
    aligned_str = "".join(
        ["|" if s1 == s2 else " " for s1, s2 in zip(seq1, seq2)])
    blosum_score = sum([BLOSUM62[s1][s2] for s1, s2 in zip(seq1, seq2)])
    return aligned_str, aligned_str.count("|"), blosum_score

class AminoAcids(Enum):
    polar = [*"STYNQ"]
    non_polar = [*"GAVCPLIMWF"]
    pos_charged = [*"KRH"]
    neg_charged = [*"DE"]
    aromatic = [*"FYW"]
    special = [*"X"]