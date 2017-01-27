
MODEL=VGG16
DATASET=working

DATA_DIR=data
FEATURES_DIR=features/${DATASET}
IMAGE_DIR=${DATA_DIR}/${DATASET}

FEATURES_META_FILE=${FEATURES_DIR}/${DATASET}.meta
REPR_PCA_FILE=${FEATURES_DIR}/${DATASET}.pca
REPR_FILE=${FEATURES_DIR}/${DATASET}.repr.npy

.PHONY: all setup features repr test clean clean-features

all:

setup:
	mkdir ${DATA_DIR}
	mkdir ${DATA_DIR}/raw
	mkdir ${DATA_DIR}/interim
	mkdir database
	mkdir features
	mkdir models


features: ${FEATURES_META_FILE}

pca:
	cmd/extract_features.py --features-dir ${FEATURES_DIR} pca ${DATASET}

repr: features ${REPR_FILE}

${FEATURES_META_FILE}:
	cmd/extract_features.py --root-dir data/ --features-dir ${FEATURES_DIR} \
		--image-dir ${IMAGE_DIR} --model ${MODEL} features ${DATASET}

${REPR_FILE}:
	cmd/extract_features.py --features-dir ${FEATURES_DIR} repr ${DATASET}

test:
	cmd/test_runner.py

clean: clean-features

clean-features:
	rm -f ${FEATURES_META_FILE} ${FEATURES_DIR}/features/*


##### Datasets #####

NOTARY_CHARTERS_DIR=${DATA_DIR}/raw/notary_charters
NOTARY_CHARTERS_RESIZED_DIR=${DATA_DIR}/interim/notary_charters

dataset-notary-charters: ${NOTARY_CHARTERS_DIR}/notary_charters.csv ${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters

${NOTARY_CHARTERS_DIR}/notary_charters.csv:
	src/data/notary_charters/dl_notary_charters.py -o ${NOTARY_CHARTERS_DIR} \ 
		data/external/notary_charters/notary_charters.xml

${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters:
	src/data/resize.py ${NOTARY_CHARTERS_DIR}/notary_charters \
		${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters


dataset-working: ${NOTARY_CHARTERS_DIR}/notary_charters.csv
	mkdir -p ${DATA_DIR}/working
	src/data/working.py --data-dir ${DATA_DIR} --out-dir ${DATA_DIR}/working