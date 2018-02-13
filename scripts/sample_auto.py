# -*- coding: utf-8 -*-
#
# sample x and z for languages while keeping W fixed
#
import sys
import codecs
import json
import numpy as np
import random
import six
# from itertools import ifilter
from argparse import ArgumentParser

from json_utils import load_json_file, load_json_stream
from mda import MatrixDecompositionAutologistic
from train import create_maps, create_mat

def dumps(mda, _iter):
    return {
        "iter": _iter,
        "x": mda.mat.tolist(),
        "z": mda.zmat.tolist(),
        "v": mda.vks.tolist(),
        "h": mda.hks.tolist(),
        "a": mda.alphas.tolist(),
    }

def main():
    # sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument("-i", "--iter", dest="_iter", metavar="INT", type=int, default=10,
                        help="# of iterations")
    parser.add_argument("--a_repeat", dest="a_repeat", metavar="INT", type=int, default=1)
    # parser.add_argument("--start", metavar="INT", type=int, default=0,
    #                     help="start")
    # parser.add_argument("--end", metavar="INT", type=int, default=np.inf,
    #                     help="end (exclusive)")
    parser.add_argument("model", metavar="MODEL", default=None)
    # parser.add_argument("langs", metavar="LANGS", default=None)
    parser.add_argument("fid2struct", metavar="FLIST", default=None)
    parser.add_argument("output", metavar="OUTPUT", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    fid2struct = load_json_file(args.fid2struct)
    P, M, fmap, bmap = create_maps(fid2struct)
    spec = six.moves.cPickle.load(open(args.model, "rb"))
    mda = spec["model"]
    L, P = mda.mat.shape
    mda.fmap = fmap
    mda.bmap = bmap

    f = sys.stdout if args.output == "-" else open(args.output, "w")
    sys.stderr.write("iter 0\n")
    f.write("%s\n" % json.dumps(dumps(mda, 0)))
    mda.init_tasks(a_repeat=args.a_repeat, sample_w=False)
    for _iter in six.moves.xrange(args._iter - 1): # already have iter 0
        sys.stderr.write("iter {}\n".format(_iter + 1))
        mda.sample(_iter=_iter)
        f.write("%s\n" % json.dumps(dumps(mda, _iter + 1)))
    if not f == sys.stdout:
        f.close()

if __name__ == "__main__":
    main()
