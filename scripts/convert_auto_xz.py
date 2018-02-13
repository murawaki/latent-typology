# -*- coding: utf-8 -*-
#
# sample merger for the MDA model
#
import sys
import codecs
import json
import numpy as np
import six
from argparse import ArgumentParser

from json_utils import load_json_file, load_json_stream
from train import create_maps

def update_features(lang, fid2struct):
    binsize = 0
    for fid, fnode in enumerate(fid2struct):
        size = len(fnode["vid2label"])
        wals_id = fnode["wals_id"]
        maxv, maxvv = -1, -1
        for i in six.moves.xrange(size):
            if lang["xfreq"][binsize+i] >= maxvv:
                maxvv = lang["xfreq"][binsize+i]
                maxv = i
        lang["features_filled"][wals_id] = maxv
        binsize += size
            
def main():
    parser = ArgumentParser()
    parser.add_argument("--burnin", metavar="INT", type=int, default=0,
                        help="# of burn-in iterations")
    parser.add_argument("--interval", metavar="INT", type=int, default=1,
                        help="pick up one per # samples")
    parser.add_argument("--update", action="store_true", default=False,
                        help="update features (for MVI)")
    parser.add_argument("langs", metavar="LANG", default=None)
    parser.add_argument("fid2struct", metavar="FLIST", default=None)
    args = parser.parse_args()

    fid2struct = load_json_file(args.fid2struct)
    P, M, fmap, bmap = create_maps(fid2struct)

    langlist = []
    for lang in load_json_stream(open(args.langs)):
        lang['latitude'] = float(lang['latitude'])
        lang['longitude'] = float(lang['longitude'])
        langlist.append(lang)
    L = len(langlist)

    count = 0
    xfreq = np.zeros((L, M), dtype=np.int32)
    zfreq = None
    for langdat in load_json_stream(sys.stdin):
        sys.stderr.write("+")
        if langdat["iter"] >= args.burnin and langdat["iter"] % args.interval == 0:
            count += 1
            if zfreq is None:
                zfreq = np.zeros((L, len(langdat["z"])), dtype=np.int32)
                # sys.stderr.write("{}\n".format(zfreq.shape))
            zfreq += np.array(langdat["z"]).T
            for lid in six.moves.xrange(L):
                for p in six.moves.xrange(P):
                    j_start, T = bmap(p)
                    xfreq[lid,j_start + langdat["x"][lid][p]] += 1
    sys.stderr.write("\n")
    for lid in six.moves.xrange(L):
        langlist[lid]["count"] = count
        langlist[lid]["xfreq"] = xfreq[lid].tolist()
        langlist[lid]["zfreq"] = zfreq[lid].tolist()
        if args.update:
            update_features(langlist[lid], fid2struct)
        sys.stdout.write("%s\n" % json.dumps(langlist[lid]))

if __name__ == "__main__":
    main()
