# -*- coding: utf-8 -*-
#
# latent representations of typological features
#
import sys, os
import shutil
import codecs
import math
import json
import numpy as np
import random
import cPickle
from argparse import ArgumentParser

from json_utils import load_json_file, load_json_stream
from mda import MatrixDecompositionAutologistic

def create_maps(fid2struct):
    # fmap(j) = p, q, j_start, T
    # bmap(p) = j_start, T
    catsize = len(fid2struct)
    binsize = 0
    map_cat2bin = np.empty((catsize, 2), dtype=np.int32) # (first elem. idx, size)
    for fid, fnode in enumerate(fid2struct):
        size = len(fnode["vid2label"])
        map_cat2bin[fid] = [binsize, size]
        binsize += size
    map_bin2cat = np.empty((binsize, 2), dtype=np.int32) # (fid, idx)
    idx = 0
    for fid, fnode in enumerate(fid2struct):
        for v, flabel in enumerate(fnode["vid2label"]):
            map_bin2cat[idx] = [fid, v]
            idx += 1
    def fmap(j):
        p, q = map_bin2cat[j]
        return p, q
    def bmap(p):
        j_start, T = map_cat2bin[p]
        return j_start, T
    return catsize, binsize, fmap, bmap

def create_mat(langlist, P):
    mat = np.zeros((len(langlist), P), dtype=np.int32)
    mvs = np.zeros((len(langlist), P), dtype=np.bool_)
    for i, lang in enumerate(langlist):
        for p, (k, k2) in enumerate(zip(lang["catvect_filled"], lang["catvect"])):
            mat[i,p] = k
            if k2 < 0:
                mvs[i,p] = True
    return mat, mvs

def create_cvlist(langlist):
    cvlist = []
    for lid, lang in enumerate(langlist):
        for fid, (v1, v2) in enumerate(zip(lang["catvect"], lang["catvect_orig"])):
            if v1 == -1 and v2 != -1:
               cvlist.append((lid, fid, v2))
    return cvlist

def create_vnet(langlist):
    vnet = []
    vgroups = {}
    for i, lang in enumerate(langlist):
        if lang["genus"] not in vgroups:
            vgroups[lang["genus"]] = []
        vgroups[lang["genus"]].append(i)
    for i, lang in enumerate(langlist):
        vnet.append(np.array([i2 for i2 in vgroups[lang["genus"]] if not i2 == i], dtype=np.int32))
    return vnet

def create_hnet(langlist):
    def _distance(x1, y1, x2, y2):
        """
        Calculate a distance between 2 points
        from whose longitude and latitude
        """
        A = 6378137.0
        B = 6356752.314140
        dy = math.radians(y1 - y2)
        dx = math.radians(x1 - x2)

        if(dx < -math.pi):
            dx += 2*math.pi
        if(dx > math.pi):
            dx -= 2*math.pi

        my = math.radians((y1 + y2) / 2)
        E2 = (A**2 - B**2) / A**2
        Mnum = A * (1 - E2)
        w = math.sqrt(1 - E2 * math.sin(my)**2)
        m = Mnum / w**3
        n = A / w
        return math.sqrt((dy * m)**2 + (dx * n * math.cos(my))**2)

    hnet = []
    for i, lang1 in enumerate(langlist):
        hvect = []
        for j, lang2 in enumerate(langlist):
            if i == j: continue
            distance = _distance(float(lang1['longitude']),
                                 float(lang1['latitude']),
                                 float(lang2['longitude']),
                                 float(lang2['latitude']))
            # in 1000km
            DISTANCE_THRESHOLD = 1000000
            if distance <= DISTANCE_THRESHOLD:
               hvect.append(j)
        hnet.append(np.array(hvect, dtype=np.int32))
    return hnet

def eval_cvlist(mda):
    cor = 0
    for lid, fid, v in mda.cvlist:
        if mda.mat[lid,fid] == v:
            cor += 1
    cv_result = { "cor": cor, "total": len(mda.cvlist) }
    sys.stderr.write("\tcv\t%f\t(%d / %d)\n" % (float(cor) / len(mda.cvlist), cor, len(mda.cvlist)))
    return cv_result

def main():
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument("--init_clusters", action="store_true", default=False)
    parser.add_argument("--only_alphas", action="store_true", default=False,
                        help="autologistic: ignore v and h")
    parser.add_argument("--drop_vs", action="store_true", default=False,
                        help="autologistic: ignore h")
    parser.add_argument("--drop_hs", action="store_true", default=False,
                        help="autologistic: ignore v")
    parser.add_argument("-i", "--iter", dest="_iter", metavar="INT", type=int, default=1000,
                        help="# of iterations")
    parser.add_argument("--alpha", metavar="FLOAT", type=float, default=-1.0,
                        help="parameter alpha")
    parser.add_argument("-K", "--initK", metavar="INT", type=int, default=100,
                        help="initial K")
    parser.add_argument('--norm_sigma', type=float, default=5.0,
                        help='standard deviation of Gaussian prior for u')
    parser.add_argument('--gamma_shape', type=float, default=1.0,
                            help='shape of Gamma prior for v and h')
    parser.add_argument('--gamma_scale', type=float, default=0.001,
                            help='scale of Gamma prior for v and h')
    parser.add_argument("--maxanneal", metavar="INT", type=int, default=0)
    parser.add_argument("--cv", action="store_true", default=False,
                        help="some features are intentionally hidden (but kept as \"catvect_orig\")")
    parser.add_argument("--output", dest="output", metavar="FILE", default=None,
                        help="save the model to the specified path")
    parser.add_argument("--resume", metavar="FILE", default=None,
                        help="resume training from model dump")
    parser.add_argument("--resume_if", action="store_true", default=False,
                        help="resume training if the output exists")
    parser.add_argument("langs", metavar="LANG", default=None)
    parser.add_argument("fid2struct", metavar="FLIST", default=None)
    # parser.add_argument("model", metavar="MODEL", default=None)
    args = parser.parse_args()
    if args.alpha < 0.0:
        args.alpha = args.initK / 10.0
    sys.stderr.write("args\t{}\n".format(args))

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    fid2struct = load_json_file(args.fid2struct)
    P, M, fmap, bmap = create_maps(fid2struct)

    offset = 0
    if args.resume_if:
        if os.path.isfile(args.output + ".current"):
            args.resume = args.output + ".current"
        elif os.path.isfile(args.output + ".best"):
            args.resume = args.output + ".best"
    if args.resume:
        sys.stderr.write("loading model from %s\n" % args.resume)
        spec = cPickle.load(open(args.resume, "rb"))
        mda = spec["model"]
        mda.init_dump(fmap, bmap)
        sys.stderr.write("iter %d: %f\n" % (spec["iter"] + 1, spec["ll"]))
        if args.cv:
            eval_cvlist(mda)
        offset = spec["iter"] + 1
    else:
        langlist = []
        for lang in load_json_stream(open(args.langs)):
            langlist.append(lang)
        mat, mvs = create_mat(langlist, P)

        sys.stderr.write("building vnet\n")
        vnet = create_vnet(langlist)
        sys.stderr.write("building hnet\n")
        hnet = create_hnet(langlist)
        mda = MatrixDecompositionAutologistic(mat, M, fmap, bmap,
                                              vnet=vnet, hnet=hnet,
                                              K=args.initK, mvs=mvs,
                                              only_alphas=args.only_alphas,
                                              drop_vs=args.drop_vs,
                                              drop_hs=args.drop_hs,
                                              norm_sigma=args.norm_sigma,
                                              gamma_shape=args.gamma_shape,
                                              gamma_scale=args.gamma_scale)
        if args.cv:
            mda.cvlist = create_cvlist(langlist)
        if args.init_clusters:
            mda.init_with_clusters(args.initK)
        else:
            mda.init_with_freq(args.initK, anneal=0.01)
        sys.stderr.write("iter 0: %f\n" % (mda.calc_loglikelihood()))
        if args.cv:
            eval_cvlist(mda)
    ll_max = -np.inf
    for _iter in xrange(offset, args._iter):
        mda.sample(_iter=_iter, maxanneal=args.maxanneal)
        ll = mda.calc_loglikelihood()
        sys.stderr.write("iter %d: %f\n" % (_iter + 1, ll))
        if args.cv:
            cv_result = eval_cvlist(mda)
        if args.output is not None:
            with open(args.output + ".current", "wb") as f:
                obj = { "model": mda.dumps(), "iter": _iter, "ll": ll }
                if args.cv:
                    obj["cv_result"] = cv_result
                cPickle.dump(obj, f)
        if ll > ll_max:
            ll_max = ll
            shutil.copyfile(args.output + ".current", args.output + ".best")
    if args.output is not None:
        with open(args.output + ".final", "wb") as f:
            obj = { "model": mda.dumps(), "iter": _iter, "ll": ll }
            if args.cv:
                obj["cv_result"] = cv_result
            cPickle.dump(obj, f)

if __name__ == "__main__":
    main()
