MODEL   ?= VGG16
DATASET	?= working

DATA_ROOT_DIR = data
FEATURES_DIR  = features/${DATASET}
IMAGE_DIR    ?= ${DATA_ROOT_DIR}/${DATASET}

.PHONY: all setup features repr test clean clean-features

all:

setup:
	mkdir ${DATA_ROOT_DIR}/raw
	mkdir ${DATA_ROOT_DIR}/interim
	mkdir database
	mkdir features
	mkdir -p models/evaluation
	ln -s ../../data/ web/static/data

test:
	cmd/test_runner.py

clean: clean-features

clean-features:
	rm -f ${FEATURES_META_FILE} ${FEATURES_DIR}/features/*

##### Extraction #####

FEATURES_META_FILE = ${FEATURES_DIR}/${DATASET}.meta
REPR_FILE          = ${FEATURES_DIR}/${DATASET}.repr.npy

features: ${FEATURES_META_FILE}

pca:
	cmd/extract_features.py --features-dir ${FEATURES_DIR} pca ${DATASET}

repr: features pca ${REPR_FILE}

${FEATURES_META_FILE}:
	cmd/extract_features.py --root-dir data/ --features-dir ${FEATURES_DIR} \
		--image-dir ${IMAGE_DIR} --model ${MODEL} features ${DATASET}

${REPR_FILE}:
	cmd/extract_features.py --features-dir ${FEATURES_DIR} repr ${DATASET}

##### Datasets #####

NOTARY_CHARTERS_DIR          = ${DATA_ROOT_DIR}/raw/notary_charters
NOTARY_CHARTERS_RESIZED_DIR  = ${DATA_ROOT_DIR}/interim/notary_charters
NOTARY_CHARTERS_RESIZED_SIZE = 1000

dataset-notary-charters: 	${NOTARY_CHARTERS_DIR}/notary_charters.csv \
							${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters \
							${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters/query

${NOTARY_CHARTERS_DIR}/notary_charters.csv:
	src/data/notary_charters/dl_notary_charters.py -o ${NOTARY_CHARTERS_DIR} \ 
		data/external/notary_charters/notary_charters.xml

${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters: 
	src/data/resize.py ${NOTARY_CHARTERS_DIR}/notary_charters \
		--size ${NOTARY_CHARTERS_RESIZED_SIZE} \
		${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters

${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters/query: data/external/notary_charters/labeled_annotations/labeled_annotations.csv
	rm -rf ${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters/query
	mkdir ${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters/query
	src/data/notary_charters/make_query_dataset.py \
		--size ${NOTARY_CHARTERS_RESIZED_SIZE} \
		${NOTARY_CHARTERS_DIR}/notary_charters \
		${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters/query \
		data/external/notary_charters/labeled_annotations/labeled_annotations.csv

dataset-working: ${NOTARY_CHARTERS_DIR}/notary_charters.csv
	mkdir -p ${DATA_DIR}/working
	src/data/working.py --data-dir ${DATA_DIR} --out-dir ${DATA_DIR}/working

##### Evaluation #####

evaluate-notary-charters: ${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters/query
	cmd/evaluate.py \
		models/evaluation/config_notary_charters_${MODEL}.txt \
		${NOTARY_CHARTERS_RESIZED_DIR}/notary_charters/query/labeled_crops.csv \
		models/evaluation/predictions_notary_charters_${MODEL}.txt
