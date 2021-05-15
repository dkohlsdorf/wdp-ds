import subprocess
import shlex

import sys

def cmd_train(labels, wav, noise, out, th):
    return "python pipeline.py train {} {} {} {} {}".format(
        labels, wav, noise, out, th
    )

def cmd_test(inp, out):
    return "python pipeline.py test {} {}".format(
        inp, out
    )

def cmd_align(inp, out, th):
    return "python pipeline.py test {} {} {}".format(
        inp, out, th
    )

if __name__ == "__main__":
    if len(sys.argv) == 6:
        labels = sys.argv[1]
        wav    = sys.argv[2]
        noise  = sys.argv[3]
        out    = sys.argv[4]
        inp    = sys.argv[5] 
        for th in range(5, 100, 5):        
            for th_nw in range(5, 100, 5):    
                th_str = str(th)
                th_nw_str = str(th_nw)
                folder = "{}/{}_{}".format(out, th_str, th_nw_str)
                subprocess.call(shlex.split('mkdir {}'.format(folder)))
                subprocess.call(shlex.split(cmd_train(labels, wav, noise, folder, th)))
                subprocess.call(shlex.split(cmd_test(inp, folder)))
                subprocess.call(shlex.split('cp {}/*.wav {}'.format(inp, folder)))
                subprocess.call(shlex.split(cmd_align(folder, folder, th_nw)))
