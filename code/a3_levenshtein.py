import os
import re
import numpy as np

# dataDir = '/u/cs401/A3/data/'
dataDir = r'C:\Users\Jerry\Documents\CSC401\A3\data'
punctuation = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""
punctuation_pattern = re.compile("([" + punctuation + "])*?")

square_pattern = re.compile("\[.*?\]")
angle_patttern = re.compile("\<.*?\>")

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    n = len(r)
    m = len(h)
    R = np.zeros((n+1, m+1))
    B = np.zeros((n+1, m+1))

    # initialize distances
    for i in range(1, n + 1):
        R[i, 0] = i
    for j in range(1, m + 1):
        R[0, j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            delete = R[i-1, j] + 1
            insert = R[i, j - 1] + 1
            substitute = R[i-1, j-1] + 0 if r[i-1] == h[j-1] else R[i-1, j-1] + 1

            R[i, j] = min(delete, insert, substitute)

            if R[i, j] == delete:
                B[i, j] = 1
            elif R[i, j] == insert:
                B[i, j] = 2
            else:
                B[i, j] = 3

    accuracy = R[n, m] / n

    # convert B into a dictionary to get counts
    unique, counts = np.unique(B, return_counts=True)
    backtract_dict = dict(zip(unique, counts))
    delete = 0
    insert = 0
    substitute = 0
    if 1 in backtract_dict.keys():
        delete = backtract_dict[1]
    if 2 in backtract_dict.keys():
        insert = backtract_dict[2]
    if 3 in backtract_dict.keys():
        substitute = backtract_dict[3]

    return accuracy, delete, insert, substitute


def preproc(line):
    line = line.lower().strip()
    words = line.split()
    line = " ".join(words[2:])  # removes the name of the sentence
    line = re.sub(square_pattern, '', line)
    line = re.sub(angle_patttern, '', line)
    line = re.sub(punctuation_pattern, '', line)
    line = re.sub(' +', ' ', line)
    words = line.split()
    return words


if __name__ == "__main__":
    google_wer = []
    kaldi_wer = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            ref_lines = open(os.path.join(dataDir, speaker, 'transcripts.txt'), 'r').read().splitlines()
            google_lines = open(os.path.join(dataDir, speaker, 'transcripts.Google.txt'), 'r').read().splitlines()
            kaldi_lines = open(os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt'), 'r').read().splitlines()

            length = min(len(ref_lines), len(google_lines), len(kaldi_lines))
            for i in range(length):
                ref_line = preproc(ref_lines[i])
                google_line = preproc(google_lines[i])
                kaldi_line = preproc(kaldi_lines[i])

                wer, delete, insert, substitute = Levenshtein(ref_line, google_line)
                print('{} {} {} {} S:{}, I:{}, D:{}'.format(speaker, 'Google', i, wer, substitute,  insert, delete))
                google_wer.append(wer)

                wer, delete, insert, substitute = Levenshtein(ref_line, kaldi_line)
                print('{} {} {} {} S:{}, I:{}, D:{}'.format(speaker, 'Kaldi', i, wer, substitute,  insert, delete))
                kaldi_wer.append(wer)

    # evaluate
    google_wer = np.array(google_wer)
    kaldi_wer = np.array(kaldi_wer)

    print("Google mean: {}".format(google_wer.mean()))
    print("Google std: {}".format(google_wer.std()))
    print("Kaldi mean: {}".format(kaldi_wer.mean()))
    print("Kaldi std: {}".format(kaldi_wer.std()))

