import os
import sys
import glob
import matplotlib.pyplot as plt

from subprocess import check_output


K = 46
FLOOR = 1.0

def prepare_project(foldder, inputs, states):
    out = check_output(["rm", "-rf", folder])
    out = check_output(["mkdir", folder])
    out = check_output(["mkdir", "{}/data".format(folder)])
    out = check_output("python pipeline.py htk results {} {} {}/data {}/clusters.mlf".format(K, inputs,  folder, folder).split(" "))
    out = check_output("python pipeline.py htk hmm_proto {} {}".format(states, folder).split(" "))
    out = check_output("python pipeline.py htk gram {}/clusters_TRAIN.mlf {}/gram".format(folder, folder).split(" "))
    out = check_output("python pipeline.py htk dict {}/clusters_TRAIN.mlf {}/dict".format(folder, folder).split(" "))

    
def init_flat(folder):
    files = glob.glob("{}/data/train/*.htk".format(folder))
    out = check_output(["rm", "-rf", "{}/hmm0".format(folder)])
    out = check_output(["mkdir", "{}/hmm0".format(folder)])
    out = check_output("HCompV -v {} -A -T 10 -M {}/hmm0 -m {}/proto".format(FLOOR, folder, folder).split(" ") + files)
    out = check_output("HParse {}/gram {}/wdnet".format(folder, folder).split(" "))
    out = check_output("python pipeline.py htk mmf {}/hmm0/proto {}/clusters_TRAIN.mlf {}/hmm0/hmm_mmf {}/list".format(folder, folder, folder, folder).split(" "))
    

def get_ll(out):
    out = out.split(b"\n")
    for line in out:
        if b"average log prob per frame" in line:
            ll = line.strip().split(b" ")[-1]
            ll = float(ll)
            return ll

        
def take_step(folder, i):
    files = glob.glob("{}/data/train/*.htk".format(folder))
    out = check_output(["rm", "-rf", "{}/hmm{}".format(folder, i)])
    out = check_output(["mkdir", "{}/hmm{}".format(folder, i)])
    out = check_output("HERest -A -T 1 -v {} -I {}/clusters_TRAIN.mlf -M {}/hmm{} -H {}/hmm{}/hmm_mmf {}/list".format(FLOOR, folder, folder, i, folder, i - 1, folder).split(" ") + files)
    return get_ll(out)


def eval(folder, last_hmm):
    files = glob.glob("{}/data/test/*.htk".format(folder))
    out = check_output("HVite -T 1 -n 10 -p 0.0 -s 5.0 -H {}/hmm{}/hmm_mmf -i {}/predictions.mlf -w {}/wdnet {}/dict {}/list".format(
        folder, last_hmm, folder, folder, folder, folder
    ).split(" ") + files)
    out = check_output("HResults -I {}/clusters_TEST.mlf {}/list {}/predictions.mlf".format(folder, folder, folder).split(" "))
    return out.decode("utf-8")


if __name__ == "__main__":
    print("=====================================")
    print("HTK Training Script")
    print("by Daniel Kyu Hwa Kohlsdorf")
    folder = sys.argv[1]
    inputs = sys.argv[2]
    states = int(sys.argv[3])
    niter  = int(sys.argv[4]) 

    print("Prepare project: {}".format(folder))
    prepare_project(folder, inputs, states)
    print("... flat start")
    init_flat(folder)
    likelihoods = []
    for i in range(1, niter + 1):
        ll = take_step(folder, i)
        likelihoods.append(ll)
        print("... reest: {} {}".format(i, ll))
    result = eval(folder, niter)
    
    print(result)
    plt.plot(likelihoods)
    plt.title("Likeihood HMM Mix")
    plt.xlabel("epoch")
    plt.ylabel("ll")
    plt.savefig('{}/ll'.format(folder))
    plt.close()
    print("=====================================")
