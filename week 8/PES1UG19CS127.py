import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        s = [self.emissions_dict[i] for i in seq]
        seq_len = len(s)

        D = np.zeros((self.N, seq_len))
        E = np.zeros((self.N, seq_len-1)).astype(np.int32)
        D[:, 0] = np.multiply(self.pi, self.B[:, s[0]])

        # Compute D and E in a nested loop
        for n in range(1, seq_len):
            for i in range(self.N):
                temp_product = np.multiply(self.A[:, i], D[:, n-1])
                D[i, n] = np.max(temp_product) * self.B[i, s[n]]
                E[i, n-1] = np.argmax(temp_product)

        # Backtracking
        S_opt = np.zeros(seq_len).astype(np.int32)
        S_opt[-1] = np.argmax(D[:, -1])
        for n in range(seq_len-2, -1, -1):
            S_opt[n] = E[int(S_opt[n+1]), n]
        result = []
        for i in S_opt:
            for j in self.states_dict:
                if i == self.states_dict[j]:
                    result.append(j)
        return result
