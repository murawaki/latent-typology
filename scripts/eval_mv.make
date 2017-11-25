# -*- mode: Makefile -*-
#
# usage make -f THIS_FILE CV=10
#
TYPE_BASE_DIR := $(HOME)/research/comp-typology
TYPE_PROG_DIR := $(TYPE_BASE_DIR)/scripts

MDA_BASE_DIR := $(HOME)/research/latent-typology
MDA_PROG_DIR := $(MDA_BASE_DIR)/scripts
OUTDIR := $(MDA_BASE_DIR)/data/cv
LANGS_CVMAP_FILE := $(OUTDIR)/langs_full.cvmap.json

TYPE_DATA_DIR := $(MDA_BASE_DIR)/data
LANGS_FILE := $(TYPE_DATA_DIR)/langs_full.json
FEATURE_FILE := $(TYPE_DATA_DIR)/flist.json


MODEL_PREFIX := mda
TRAIN_OPTS := --maxanneal=100 --init_clusters

LANG_NUM := 2607 # **HARD_CODED**
LANG_BLOCK := 260
LANG_PARA_MAX := 10 # parallel: 11
LANG_XZ_ITER := 100
LANG_AUTO_ITER := 100

PYTHON := nice -19 python

CV := 10
CV_MAX := $(shell expr $(CV) - 1)
SEED := --seed=20
KS := 50 100 # 250 500 1000
ITER := 500 # 1000

NPB_KS := 50 100 250 500

$(LANGS_CVMAP_FILE) : $(LANGS_FILE)
	mkdir -p $(OUTDIR) && \
	$(PYTHON) $(TYPE_PROG_DIR)/mv/cv/make_cvmap.py $(SEED) $(LANGS_FILE) $(FEATURE_FILE) $(LANGS_CVMAP_FILE) $(CV)


# cv_split FILE_PREFIX CV_IDX
define cv_main
$(1).cv$(2).json : $(LANGS_FILE) $(LANGS_CVMAP_FILE)
	$(PYTHON) $(TYPE_PROG_DIR)/mv/cv/hide.py $(LANGS_FILE) $(1).cv$(2).json $(LANGS_CVMAP_FILE) $(2)

HIDE_LIST += $(1).cv$(2).json

$(1).cv$(2).tsv : $(1).cv$(2).json
	$(PYTHON) $(TYPE_PROG_DIR)/mv/json2tsv.py $(1).cv$(2).json $(FEATURE_FILE) $(1).cv$(2).tsv

$(1).cv$(2).filled.tsv : $(1).cv$(2).tsv
	R --vanilla -f $(TYPE_PROG_DIR)/mv/impute_mca.r --args $(1).cv$(2).tsv $(1).cv$(2).filled.tsv

$(1).cv$(2).filled.json : $(1).cv$(2).filled.tsv
	$(PYTHON) $(TYPE_PROG_DIR)/mv/tsv2json.py $(1).cv$(2).json $(1).cv$(2).filled.tsv $(FEATURE_FILE) $(1).cv$(2).filled.json.tmp && \
	$(PYTHON) $(TYPE_PROG_DIR)/format_wals/catvect.py $(1).cv$(2).filled.json.tmp $(FEATURE_FILE) $(1).cv$(2).filled.json && \
	rm -f $(1).cv$(2).filled.json.tmp

FILLED_LIST += $(1).cv$(2).filled.json
endef


$(foreach i,$(shell seq 0 $(CV_MAX)), \
  $(eval $(call cv_main,$(OUTDIR)/langs_full,$(i))))


# cv_main_npb FILE_PREFIX CV_IDX K
define cv_main_npb
$(1).npb$(3).cv$(2).filled.json : $(1).cv$(2).tsv
	mkdir -p $(1).npb$(3).cv$(2).tmp \
	&& R --vanilla -f $(TYPE_PROG_DIR)/mv/impute_npb.r --args $(1).cv$(2).tsv $(1).npb$(3).cv$(2).tmp/imputed $(3) > $(1).npb$(3).cv$(2).log \
	&& python $(TYPE_PROG_DIR)/mv/tsv2json_merge.py $(1).cv$(2).json $(1).npb$(3).cv$(2).tmp/imputed $(FEATURE_FILE) $(1).npb$(3).cv$(2).filled.json

FILLED_LIST += $(1).npb$(3).cv$(2).filled.json
NPB_LIST += $(1).npb$(3).cv$(2).filled.json
NPB$(3)_LIST += $(1).npb$(3).cv$(2).filled.json
endef

$(foreach k,$(NPB_KS), \
  $(foreach i,$(shell seq 0 $(CV_MAX)), \
    $(eval $(call cv_main_npb,$(OUTDIR)/langs_full,$(i),$(k)))))

# cv_split FILE_PREFIX K
define cv_npb_eval
$(1).nbp$(2).eval : $(NPB$(2)_LIST)
	sh -c 'for i in `seq 0 $(CV_MAX)`; do $(PYTHON) $(TYPE_PROG_DIR)/mv/cv/eval_mv.py $(SEED) $(LANGS_FILE) $(1).npb$(2).cv$$$${i}.filled.json -; done | perl -anle"\$$$$a+=\$$$$F[1];\$$$$b+=\$$$$F[2]; END{printf \"%f\\t%d\\t%d\\n\", \$$$$a / \$$$$b, \$$$$a, \$$$$b;}" > $(1).nbp$(2).eval'

EVALS += $(1).nbp$(2).eval
EVALS_NPB += $(1).nbp$(2).eval
endef

$(foreach k,$(NPB_KS), \
  $(eval $(call cv_npb_eval,$(OUTDIR)/langs_full,$(k))))





$(OUTDIR)/langs_full.random.eval : $(LANG_FILE)
	mkdir -p $(OUTDIR)
	$(PYTHON) $(TYPE_PROG_DIR)/mv/cv/eval_mv.py $(SEED) --random $(LANGS_FILE) - $(FEATURE_FILE) > $(OUTDIR)/langs_full.random.eval
EVALS += $(OUTDIR)/langs_full.random.eval

$(OUTDIR)/langs_full.freq.eval : $(LANG_FILE) $(HIDE_LIST)
	sh -c 'for i in `seq 0 $(CV_MAX)`; do $(PYTHON) $(TYPE_PROG_DIR)/mv/cv/eval_mv.py $(SEED) --freq $(LANGS_FILE) $(OUTDIR)/langs_full.cv$${i}.json $(FEATURE_FILE); done | perl -anle"\$$a+=\$$F[1];\$$b+=\$$F[2]; END{printf \"%f\\t%d\\t%d\\n\", \$$a / \$$b, \$$a, \$$b;}" > $(OUTDIR)/langs_full.freq.eval'

EVALS += $(OUTDIR)/langs_full.freq.eval

$(OUTDIR)/langs_full.mcr.eval : $(LANG_FILE) $(FILLED_LIST)
	sh -c 'for i in `seq 0 $(CV_MAX)`; do $(PYTHON) $(TYPE_PROG_DIR)/mv/cv/eval_mv.py $(SEED) $(LANGS_FILE) $(OUTDIR)/langs_full.cv$${i}.filled.json -; done | perl -anle"\$$a+=\$$F[1];\$$b+=\$$F[2]; END{printf \"%f\\t%d\\t%d\\n\", \$$a / \$$b, \$$a, \$$b;}" > $(OUTDIR)/langs_full.mcr.eval'

EVALS += $(OUTDIR)/langs_full.mcr.eval



# cv_split FILE_PREFIX K CV_IDX
define cv_mda_train
$(1)_K$(2)_cv$(3).pkl : $(OUTDIR)/langs_full.cv$(3).filled.json
	$(PYTHON) $(MDA_PROG_DIR)/train.py $(SEED) --iter=$(ITER) --cv --initK=$(2) --output=$(1)_K$(2)_cv$(3).pkl --resume_if $(OUTDIR)/langs_full.cv$(3).filled.json $(TRAIN_OPTS) $(FEATURE_FILE) >> $(1)_K$(2)_cv$(3).log 2>&1 && cp $(1)_K$(2)_cv$(3).pkl.final $(1)_K$(2)_cv$(3).pkl

MDAS += $(1)_K$(2)_cv$(3).pkl
endef

$(foreach k,$(KS), \
  $(foreach i,$(shell seq 0 $(CV_MAX)), \
    $(eval $(call cv_mda_train,$(OUTDIR)/$(MODEL_PREFIX),$(k),$(i)))))




# cv_split FILE_PREFIX K CV_IDX
define cv_mda_sample_auto
$(1)_K$(2)_cv$(3).xz.json.bz2 : $(1)_K$(2)_cv$(3).pkl
	$(PYTHON) $(MDA_PROG_DIR)/sample_auto.py $(SEED) --a_repeat=5 --iter=$(LANG_AUTO_ITER) $(1)_K$(2)_cv$(3).pkl $(FEATURE_FILE) - | bzip2 -c > $(1)_K$(2)_cv$(3).xz.json.bz2

SAMPLE_AUTO += $(1)_K$(2)_cv$(3).xz.json.bz2
SAMPLE_XZ_$(2) += $(1)_K$(2)_cv$(3).xz.json.bz2
SAMPLE_XZ_$(2)_$(3) += $(1)_K$(2)_cv$(3).xz.json.bz2

$(1)_K$(2)_cv$(3).xz.merged.json : $(1)_K$(2)_cv$(3).xz.json.bz2
	bzcat $(1)_K$(2)_cv$(3).xz.json.bz2 | $(PYTHON) $(MDA_PROG_DIR)/convert_auto_xz.py --burnin=0 --update $(OUTDIR)/langs_full.cv$(3).filled.json $(FEATURE_FILE) > $(1)_K$(2)_cv$(3).xz.merged.json

SAMPLE_XZ_MERGED += $(1)_K$(2)_cv$(3).xz.merged.json
SAMPLE_XZ_MERGED_$(2) += $(1)_K$(2)_cv$(3).xz.merged.json
endef


# cv_split FILE_PREFIX K
define cv_mda_eval
$(1)_K$(2).eval : $(SAMPLE_XZ_MERGED_$(2))
	sh -c 'for i in `seq 0 $(CV_MAX)`; do $(PYTHON) $(TYPE_PROG_DIR)/mv/cv/eval_mv.py $(SEED) $(LANGS_FILE) $(1)_K$(2)_cv$$$${i}.xz.merged.json -; done | perl -anle"\$$$$a+=\$$$$F[1];\$$$$b+=\$$$$F[2]; END{printf \"%f\\t%d\\t%d\\n\", \$$$$a / \$$$$b, \$$$$a, \$$$$b;}" > $(1)_K$(2).eval'

EVALS += $(1)_K$(2).eval
endef


$(foreach k,$(KS), \
  $(foreach i,$(shell seq 0 $(CV_MAX)), \
     $(eval $(call cv_mda_sample_auto,$(OUTDIR)/$(MODEL_PREFIX),$(k),$(i)))))
$(foreach k,$(KS), \
  $(eval $(call cv_mda_eval,$(OUTDIR)/$(MODEL_PREFIX),$(k))))


all : $(EVALS)

npb : $(EVALS_NPB)

filled : $(FILLED_LIST)

mda : $(MDAS)

clean :
	rm -f $(OUTDIR)/langs*
	rmdir --ignore-fail-on-non-empty $(OUTDIR)

.PHONY : all clean npb filled

sample_xz_merged : $(SAMPLE_XZ_MERGED)
sample_xz_merged_50 : $(SAMPLE_XZ_MERGED_50)
sample_xz_merged_100 : $(SAMPLE_XZ_MERGED_100)

eval_50 : $(OUTDIR)/$(MODEL_PREFIX)_K50.eval
eval_100 : $(OUTDIR)/$(MODEL_PREFIX)_K100.eval
eval_250 : $(OUTDIR)/$(MODEL_PREFIX)_K250.eval
eval_500 : $(OUTDIR)/$(MODEL_PREFIX)_K500.eval
