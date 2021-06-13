import numpy as np
import struct


PERIOD      = 1
SAMPLE_SIZE = 4 
USER        = 9
ENDIEN      = 'big'


def write_htk(sequence, to):
    n = len(sequence)
    dim = len(sequence[0])
    with open(to, "wb") as f:
        sze = dim * SAMPLE_SIZE
        f.write(n.to_bytes(4, byteorder=ENDIEN))
        f.write(PERIOD.to_bytes(4, byteorder=ENDIEN))
        f.write(sze.to_bytes(2, byteorder=ENDIEN))
        f.write(USER.to_bytes(2, byteorder=ENDIEN))
        for i in range(0, n):
            for j in range(0, dim):
                x = sequence[i, j]
                ba = bytearray(struct.pack(">f", x))
                f.write(ba)

                
def vec(v):    
    line = " ".join([str(x) for x in v])
    return "\t{}".format(line)


def mat(x):
    return "\n".join([vec(x[i]) for i in range(len(x))])


def left_right_hmm(max_states, min_states, dims, name="proto"):
    transitions = np.zeros((max_states, max_states))    
    means       = np.zeros(dims)
    variances   = np.ones(dims)  

    transitions[0, 1] = 1.0
    states = []
    for i in range(1, max_states - 1):
        state = """
        <State> {}
          <Mean> {}
           {}
          <Variance> {}
           {}
        """.format(i + 1, dims, vec(means), dims, vec(variances))
        states.append(state)
        transitions[i,i] = 0.9
        if i == min_states:
            transitions[i, i + 1] = 0.05
            transitions[i, max_states - 1] = 0.05
        elif i + 1 < max_states:
            transitions[i, i + 1] = 0.1

    return """
    ~o <VecSize> {} <USER>
    ~h "{}"

    <BeginHMM>
      <NumStates> {}
      {} 
      <TransP> 8
      {}
    <EndHMM>
    """.format(dims, name, max_states, "".join(states), mat(transitions))


def mmf(label_file, proto_file, hmm_out="hmm0", hmm_list_out="monophones"):
    df = pd.read_csv(label_file, sep=" ", header=None, names=["start", "stop", "lab"], skiprows=2)
    df = df.dropna()
    labels = set(df["lab"])
    print(labels)
    lines = [line for line in open(proto_file)]

    hmm = []
    start = False
    for line in lines:
        if line.startswith("<BEGINHMM>"):
            start = True
        if start:
            hmm.append(line)    
        if line.endswith("<ENDHMM>"):
            break

    monophones = []
    mmf = ["""~o <VECSIZE> 128 <USER><DIAGC>"""]

    for i in labels:    
        header = "~h \"{}\"\n".format(i)
        monophones.append("{}".format(i))
        mmf.append(header)
        mmf.extend(hmm)
        mmf.append("\n")

    mmf = "".join(mmf)
    monophones = "\n".join(monophones)
    with open(hmm_out, "w") as f:
        f.write(mmf)
    with open(hmm_list_out, "w") as f:
        f.write(monophones)        
