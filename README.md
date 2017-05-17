# Historic Image Retrieval using Deep Learned Features

This repository contains the implementation of an image retrieval system. 
The project was implemented using Python, Tensorflow and Keras. 
The implementation follows [this paper](tolias15) from Tolias et al, who make use of global image descriptors obtained from Convolutional Neural Networks. 
For ease of testing, the project also includes a simple web frontend which allows to query the extracted image descriptor database interactively.

The system was designed for the application of finding recurring occurences of patterns in historical documents, e.g. a certain sign. 
A working system could potentially simplify the life of historians, who currently have to sift through heaps of scanned documents by hand.

## Setup

#### Prerequisites:

- Get all dependencies using [Conda](https://conda.io/): `conda env create -n img-retrieval -f environment.yml`
- Activate the environment: `source activate img-retrieval`
- Setup the folder structure: `make setup`
- Download the notary charters dataset: `make dataset-notary-charters`

#### Building the image descriptor database:

```DATASET=notary_charters IMAGE_DIR=data/interim/notary_charters/notary_charters make repr```

Here, `DATASET` contains the name referring to the generated dataset, and `IMAGE_DIR` is the directory containing the images used for the image database. 
It might take a while until the process is finished.

#### Query the image database using the web frontend:

WIP

- Setup the metadata database: cmd/modify_database.py --root-dir data/ create database/notary_charters.db data/raw/notary_charters/
TODO: verify this command

- Start the web server on localhost:
```
cd web
./main.py
```
TODO: fix server config

- You can access the web frontend on localhost:5000
TODO: verify

#### Evaluating the performance of the retrieval system:

```make evaluate-notary-charters```

The evaluation uses hand-labeled pattern occurences, which is why it is only available for the notary charters dataset. 
The evaluation scripts reports:
- retrieval performance as mAP, i.e. the mean average precision over all correctly retrieved images
- localization performance as mAP, i.e. the mean average precision over all correctly retrieved and localized images
- localization performance as IoU, i.e. the intersection over union score of all correctly retrieved images


## Code Structure


## References

- [Giorgos Tolias, Ronan Sicre, Hervé Jégou - Particular object retrieval with integral max-pooling of CNN activations](tolias15)



[tolias15]: https://arxiv.org/abs/1511.05879