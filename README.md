# Animal Classification Model

<div align="center">
  <img src="https://img.shields.io/github/stars/mlops-hub/classifier-model.svg?style=for-the-badge" alt="GitHub Stars" />
  <img src="https://img.shields.io/github/forks/mlops-hub/classifier-model.svg?style=for-the-badge" alt="GitHub Forks" />
  <img src="https://img.shields.io/github/contributors/mlops-hub/classifier-model.svg?style=for-the-badge" alt="Contributors" />
  <img src="https://img.shields.io/github/last-commit/mlops-hub/classifier-model/main.svg?style=for-the-badge" alt="Last Commit" />
  <img src="https://img.shields.io/badge/python-3.12.x-blue?style=for-the-badge" alt="Python Version" />
</div>

<hr />


## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Model Workflow](#model-workflow)
- [Setup & Installation](#setup--installation)
    - [Clone repo](#step-1-clone-the-repository)
    - [Create Virtual Environment](#step-2-create-virtual-environment)
    - [Install Dependencies](#step-3-install-dependencies)
    - [Training the Model](#step-4-training-the-model)
    - [Testing/Prediction](#step-5-testingprediction)
- [Contribution](#contribution)
- [References](#references)


## Overview

This project is an **"Animal Classification System"** built using machine learning. It predicts the class of an animal (e.g., Mammal, Bird, Fish, etc.) based on its features. The model is trained on the UCI Zoo dataset and related class information


## Datasets

[UCI Zoo Dataset](https://archive.ics.uci.edu/dataset/111/zoo)

[Kaggle Zoo Animal Classification](https://www.kaggle.com/datasets/uciml/zoo-animal-classification/data)


The main dataset used is [zoo_animals_data.csv](./zoo_animals_data.csv), which contains features such as:


| Feature      | Description           |
|--------------|-----------------------|
| hair         | Has hair              |
| feathers     | Has feathers          |
| eggs         | Lays eggs             |
| milk         | Produces milk         |
| airborne     | Can fly/airborne      |
| aquatic      | Lives in water        |
| predator     | Is a predator         |
| toothed      | Has teeth             |
| backbone     | Has backbone          |
| breathes     | Breathes air          |
| venomous     | Is venomous           |
| fins         | Has fins              |
| legs         | Number of legs        |
| tail         | Has tail              |
| domestic     | Is domesticated       |
| catsize      | Cat-sized             |
| class_type   | Class type (numeric)  |
| class_name   | Class name (label)    |


## Model Workflow

<img src="./assets//classifier-arch-new.jpg" alt="model workflow" />


## Project Structure

```bash

classifier_model
|__ venv/                              # virtual env
|__ datasets/
    |__ zoo_data.csv                   # original zoo dataset
    |__ class.csv                      # original class name dataset
    |__ final_dataset.csv              # final dataset for model
|__ feature_store/
    |__ preprocessed_data.csv          # save preprocessed data for testing
    |__ feature_names.pkl              # save feature_names for testing
|__ models/
    |__ classifier_model.pkl           # saved model in .pkl
|__ logs/*                             # logs for hyperparamter tuning values
|__ src/
  |__ index.py                         # main file to run
  |__ data_piepline/                   # data_pipeline folder
      |__ data_*.py
      |__ index.py
  |__ model_pipeline/                  # model_pipeline folder
      |__ mdoel_*.py
      |__ index.py
  |__ test_model.py                    # to test model
|__ requirements.txt                   # install dependency pacakges
|__ README.md     
                 
```


## Libraries and Tools

- **Machine Learning**: scikit-learn
- **Type of Machine Learning**: Supervised ML
- **Visual Charts**: matplotlib, seaborn
- **data validation**: pandera
- **Save model**: joblib


## Model

- **Algorithm**: Logistic Regression 
- **Evaluation**: Accuracy, classification report, confusion matrix
- **Output**: Predicted animal class



## Setup & Installation

#### Step-1: Clone the repository

```bash
git clone https://github.com/mlops-hub/classifier-model.git
cd classifier-model
```

#### Step-2: Create Virtual Environment

##### Windows

```bash
python -m venv venv
venv\Scripts\activate
```
##### Mac and Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
#### Step-3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step-4: Run the Model

Run [`index.py`](./src/index.py) to load data, preprocess data, train and save the model.

```bash
cd src
python index.py
```

**Flow of Code Run**

```bash
index.py -> data_pipeline.py     ->  model_pipeline.py
              |__ data_ingestion       |__ model_train
              |__ data_validation      |__ model_evaluation
              |__ data_eda             |__ model_validation
              |__ feature_engg         |__ model_tuning
              |__ data_preprocess


#### Step-5: Testing/Prediction

Run [`test_model.py`](./src/test_model.py) to make predictions. If 'animal' is not found, you will be prompted to enter animal features, and the model will predict the class:

```bash
cd src
python test_model.py
```

The model achieves high accuracy in classifying animals into their respective classes.


## Contribution

Please read our [Contributing Guidelines](CONTRIBUTION.md) before submitting pull requests.


## License
This project is under [MIT Licence](LICENCE) support.
