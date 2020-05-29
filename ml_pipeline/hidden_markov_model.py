class HiddenMarkovModel:
    
    def __init__(self, transitions, observations):
        self.transitions = transitions
        self.observations = observations

    @property
    def n_states(self):
        return self.transitions.n_states  
