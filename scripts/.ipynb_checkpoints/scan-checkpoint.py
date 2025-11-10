import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

### ========================================
def Motif2Seqs(motif, mode='DNA'):
    '''
    transfer a motif to all possible sequences
    ex: 'ANCG' -> ['AACG', 'ACCG', 'AGCG', 'AUCG']
    '''
    def IUPAC_nucleotide(var, mode='DNA'):
        if mode == 'RNA':
            return {'Z': '', 
                    'A': 'A', 
                    'C': 'C', 
                    'G': 'G', 
                    'T': 'U', 
                    'U': 'U', 
                    'W': 'AU', 
                    'S': 'CG', 
                    'M': 'AC', 
                    'K': 'GU', 
                    'R': 'AG', 
                    'Y': 'CU', 
                    'B': 'CGU', 
                    'D': 'AGU', 
                    'H': 'ACU', 
                    'V': 'ACG', 
                    'N': 'ACGU', 
                    'n': 'ACGU'}.get(var, var)
        elif mode == 'DNA':
            return {'Z': '', 
                    'A': 'A', 
                    'C': 'C', 
                    'G': 'G', 
                    'T': 'T', 
                    'U': 'T', 
                    'W': 'AT', 
                    'S': 'CG', 
                    'M': 'AC', 
                    'K': 'GT', 
                    'R': 'AG', 
                    'Y': 'CT', 
                    'B': 'CGT', 
                    'D': 'AGT', 
                    'H': 'ACT', 
                    'V': 'ACG', 
                    'N': 'ACGT', 
                    'n': 'ACGT'}.get(var, var)
    
    pools = [tuple(pool) for pool in [IUPAC_nucleotide(m, mode) for m in motif]]
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield ''.join(prod)


### ========================================
def GBI_numba(arr_seq, arr_zeros, positions, SeqLen):
    L = len(positions)
    for i, p in enumerate(positions):
        arr_zeros += arr_seq//4**(SeqLen-1-p)%4*4**(L-1-i)
    return arr_zeros

def groupby_index(positions, SeqLen):
    arr_seq = np.arange(4**SeqLen)
    arr_zeros = np.zeros(4**SeqLen)
    return GBI_numba(arr_seq, arr_zeros, np.array(positions), SeqLen).astype(int)

def PWM2Score(matrix):
    '''
    Computes scores for all possible sequences generated from a position weight matrix

    Args:
        matrix (pandas.DataFrame): a position weight matrix

    Returns:
        dict: a dictionary of sequence to score pairs
    '''
    n = len(matrix)
    scores = np.array([matrix.values[i][groupby_index([i], n)] for i in range(n)]).sum(axis=0)
    return pd.DataFrame(scores, index=list(Motif2Seqs('N'*n))).to_dict()[0]

def Matrix2Score(matrix):
    scores = np.array([matrix.values[i][groupby_index([i], 6)] for i in range(6)]).sum(axis=0)
    return pd.DataFrame(scores, index=list(Motif2Seqs('N'*6))).to_dict()[0]

### ========================================
# def Params2PSAM(Params):
#     SeqLen = int(len(Params)/3)
#     Params = Params.reshape(SeqLen, 3)
#     Params = np.concatenate((Params, -Params.sum(axis=1).reshape(SeqLen, 1)), axis=1)
#     return pd.DataFrame(Params, columns=['A', 'C', 'G', 'T'])

# def LoadModel(filename):
#     popt = np.loadtxt(filename, delimiter=',')
#     SpacerPenalty = popt[12*3:12*3+5]
#     PSAM10 = Params2PSAM(popt[:6*3])
#     PSAM35 = Params2PSAM(popt[6*3:12*3])
#     popt[-1] += SpacerPenalty[2]
#     SpacerPenalty -= SpacerPenalty[2]
#     SpacerPenalty = pd.DataFrame(SpacerPenalty, index=np.linspace(19, 15, 5).astype(int))
#     ddGSP = SpacerPenalty[0].to_dict()
#     ddG10 = PWM2Score(PSAM10)
#     ddG35 = PWM2Score(PSAM35)
#     Parameters = popt[-3:]
#     return ddG10, ddG35, ddGSP, Parameters

### ========================================
# def MaximizeScore(seq, dict_score):
#     ss = window(seq, 6)
#     scores = np.array([dict_score[s] for s in ss])
#     idx = scores.argmax()
#     score_max = scores[idx]
#     seq_match = seq[idx:idx+6]
#     return idx, score_max, seq_match

# class Alignment:
#     def __init__(self, seqs, dict_score, progress=False):
#         if progress:
#             result = [MaximizeScore(seq, dict_score) for seq in tqdm(seqs)]
#         else:
#             result = [MaximizeScore(seq, dict_score) for seq in seqs]
#         self.site = np.array([i[0] for i in result])
#         self.score = np.array([i[1] for i in result])
#         self.seq = np.array([i[2] for i in result])

# ### ========================================
# def MinimizeEnergy(seq, ddG10, ddG35, SpacerPenalty, sp_max=19, sp_min=15):
#     ss = window(seq, 6)
#     ddG10_seq = np.array([ddG10[s] for s in ss[sp_max+6:]])
#     ddG35_seq = np.array([ddG35[s] for s in ss[:-sp_min-6]])
#     ddGtotal_seq = np.array([ddG10_seq[i]+np.min(ddG35_seq[i:i+(sp_max-sp_min+1)]+SpacerPenalty) for i in range(len(ddG10_seq))])
#     ##
#     site = ddGtotal_seq.argmin()
#     energy = ddGtotal_seq[site]
#     seq10 = ss[sp_max+6+site]
#     site2 = (ddG35_seq[site:site+(sp_max-sp_min+1)]+SpacerPenalty).argmin()
#     spacer = sp_max-site2
#     seq35 = ss[site+site2]
#     ##
#     return sp_max+6+site-len(seq), energy, seq10, seq35, spacer

# class Alignment_psam:
#     def __init__(self, seqs, ddG10, ddG35, SpacerPenalty):
#         result = [MinimizeEnergy(s, ddG10, ddG35, SpacerPenalty) for s in tqdm(seqs)]
#         self.site = np.array([i[0] for i in result])
#         self.energy = np.array([i[1] for i in result])
#         self.seq10 = np.array([i[2] for i in result])
#         self.seq35 = np.array([i[3] for i in result])
#         self.spacer = np.array([i[4] for i in result])

### ========================================
### ========================================
### ========================================
class MatrixScan:
    """
    To scan sequence to find the most possible motif by a position weight matrix.
    The class provides two modes:
    1. only one hexamer
    2. two hexamer and spacer length (i.e. spacer penalty)

    the format of input:
    sequences = [seq1, seq2, ..., seqN] (list)
    matrix = PFM(['TTGACATATAAT']) (dataframe)
    spacer_score = {15: -2, 16: -1, 17: 0, 18: -1, 19: -2} (dictionary)
    """
    def __init__(self, sequences, matrix, spacer_score=None):
        
        if len(matrix) == 6:

            self.matrix = PWM2Score(matrix)

            result = np.array([self.scan_minus10(s) for s in tqdm(sequences)])
            result = pd.DataFrame(result, columns=['minus10', 'score', 'site'])
            self.result = result.astype({'minus10': str, 'score': float, 'site': int})
            
            # streamline the variables
            self.matrix = matrix

        elif (len(matrix) == 12) & (spacer_score != None):

            self.spacer_score = spacer_score
            self.matrix35 = PWM2Score(matrix[:6])
            self.matrix10 = PWM2Score(matrix[6:])

            result = np.array([self.scan_hexamers(s) for s in tqdm(sequences)])
            result = pd.DataFrame(result, columns=['minus35', 'minus10', 'spacer_length', 'site10'])
            self.result = result.astype({'minus35': str, 'minus10': str, 'spacer_length': int, 'site10': int})
            
            # # streamline the variables
            # self.matrix35 = matrix[:6]
            # self.matrix10 = matrix[6:]

        else:
            print('The format of parameters is wrong!')

        return None

    def scan_minus10(self, sequence):

        ss = self.sliding_window(sequence, 6)
        arr_score = np.array([self.matrix.get(s, 0) for s in ss])

        site = arr_score.argmax()
        score = arr_score[site]
        motif = ss[site]

        return motif, score, site

    def scan_hexamers(self, sequence):

        ss = self.sliding_window(sequence, 6)
        n_win = len(ss)
        arr_score35 = np.array([self.matrix35.get(s, -10) for s in ss])
        arr_score10 = np.array([self.matrix10.get(s, -10) for s in ss])
        
        indices = []
        scores = {}
        
        # for spacer in [0]:
        #     distance = 6+16
        #     idx10 = ss[distance:n_win]
        #     indices += [["------", b, spacer, c] for b, c in zip(idx10, range(distance, n_win))]
        #     scores[spacer] = arr_score10[distance:n_win]
        
        for spacer in self.spacer_score.keys():
            distance = 6+spacer
            idx35 = ss[:(n_win-distance)]
            idx10 = ss[distance:n_win]
            indices += [[a, b, spacer, c] for a, b, c in zip(idx35, idx10, range(distance, n_win))]
            scores[spacer] = arr_score35[:(n_win-distance)] + arr_score10[distance:n_win] + self.spacer_score[spacer]
        
        idx = np.argmax(np.concatenate(list(scores.values())))
        return indices[idx]

    def sliding_window(self, string, window_size):
        """
        Generates a sliding window of a specified size for a given string.

        Args:
        string (str): The input string.
        window_size (int): The size of the sliding window.

        Returns:
        List[str]: A list of strings representing the sliding window.
        """
        window = []
        for i in range(len(string) - window_size + 1):
            window.append(string[i:(i+window_size)])

        return window

    def score_m35(self, seq35):
        return self.matrix35[seq35]

    def score_m10(self, seq10):
        return self.matrix10[seq10]
