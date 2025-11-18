"""
===============================================================================
Title:      Mutation
Outline:    Mutation class to handle mutations in protein sequences. So far, it
            only supports the validation and application of simple mutations
            in a given sequence (e.g. M13A, GS55T)
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
===============================================================================
"""

# Built-in modules
import re

class Mutation:

    def __init__(self):
        pass

    @staticmethod
    def mutate(seq: str, mutations: list[str]) -> str:
        '''
        Applies a list of mutations to a given protein sequence.

        Parameters
        ----------
        seq : str
            Initial WT protein sequence.
        mutations : list[str]
            List of mutations to apply to the sequence. Each mutation should
            be in the format 'WT_index_MUT', where WT is the wild type amino
            acid, index is the position (1-based) of the mutation, and MUT is
            the mutated amino acid.

        Returns
        -------
        str
            Mutated protein sequence.

        Raises
        ------
        ValueError
            If a mutation is not valid for the given sequence.
        '''
        for mutation in mutations:
            # Parse mutation
            wt, index, mut = re.findall(r'^([A-Z]*)([0-9]*)([A-Z]*)', mutation)[0]
            # Validate mutation
            if seq[int(index) - 1 : int(index) - 1 + len(wt)] != wt:
                raise ValueError(f'Mutation {mutation} is not valid for sequence {seq}') 
            # Mutate
            before = seq[:int(index) - 1]
            after = seq[int(index) + len(wt) - 1:]
            seq = before + mut + after

        return seq
    
if __name__ == '__main__':
    '''Test class'''
    seq = 'MGRGKIEIKRIENSTNRQVTFSKRRNGILKKAREISVLCDAEVGVVVFSSAGKLYDYCSPKTSLSKILEKYQTNSGKILWGEKHKSLSAEIDRIKKENDTMQIELRHLKGEDLNSLQPKDLIMIEEALDNGLTNLNEKLMEHWERRVTNTKMMEDENKLLAFKLHQQDIALSGSMRELELGYHPDRDLAAQMPITFRVQPSHPNLQENN'
    mutations = ['M1']
    print(Mutation.mutate(seq, mutations))


