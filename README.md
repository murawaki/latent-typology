**Now superseded by https://github.com/murawaki/lattyp**

# About

  Yugo Murawaki. 2017.  Diachrony-aware Induction of Binary Latent
  Representations from Typological Features.  In Proceedings of the
  8th International Joint Conference on Natural Language Processing
  (IJCNLP 2017), pp. 451-461.
  http://aclweb.org/anthology/I17-1046


# Requirements

- Python2
    - numpy
    - scipy

The code also depends on [the comp-typology package](https://github.com/murawaki/comp-typology).



# Inputs

- data/langs_full.json: languages taken from WALS (one language per line)
- data/flist.json: a subset of WALS features (single JSON object)

These files were generated using the comp-typology package.


# Inference

## Train the model

```
python train.py --seed=10 --initK=50 --maxanneal=100 --init_clusters --norm_sigma=10.0 --gamma_scale=1.0 --resume_if --output ../data/mda_K50.pkl ../data/langs_full.json ../data/flist.json 2>&1 | tee -a ../data/mda_K50.log
```

## Collect samples while keeping W fixed

```
python sample_auto.py --seed=10 --iter=100 --a_repeat=5 ../data/mda_K50.pkl.final ../data/flist.json  ../data/mda_K50.xz.json 2>&1 | tee  ../data/mda_K50.xz.log
python convert_auto_xz.py --burnin=0 --update ../data/mda_K50.xz.json ../data/langs_full.json ../data/flist.json > ../data/mda_K50.xz.merged.json
```


## Missing value imputation

```
make -j -f eval_mv.make all MODEL_PREFIX=mda TRAIN_OPTS="--init_clusters --maxanneal=100 --norm_sigma=10.0 --gamma_scale=1.0"

```
