# Manage your emails with Python 
[![Python package](https://github.com/pyscioffice/pydatamail_ml/actions/workflows/unittest.yml/badge.svg?branch=main)](https://github.com/pyscioffice/pydatamail_ml/actions/workflows/unittest.yml)
[![Coverage Status](https://coveralls.io/repos/github/pyscioffice/pydatamail_ml/badge.svg?branch=main)](https://coveralls.io/github/pyscioffice/pydatamail_ml?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The `pydatamail_ml` python module extends the functionality of `pydatamail` by adding support for machine learning. This
includes but is not limited to text preprocessing utilities as well as basic machine learning models based on decision
trees.

# Installation 
Install the `pydatamail_ml` package using `pip`:
```
pip install pydatamail_ml
```
Or alternatively using `conda`: 
```
conda install -c conda-forge pydatamail_ml
```

# Python interface 
Import the `pydatamail_ml` module 
```
from pydatamail_ml import one_hot_encoding, get_machine_learning_database, gather_data_for_machine_learning, train_model, get_machine_learning_recommendations, text_pipeline, detect_language
```

The individual components are briefly explained below: 

* `one_hot_encoding()` - Encoding the email headers fields (`to`, `from`, `cc`) in binary descriptors.
* `get_machine_learning_database()` - Get a database to store machine learning models in. 
* `get_gather_data_for_machine_learning()` - Clean up encoded data by separating input and output.
* `train_model()` - Train a randomforest machine learning model.
* `get_machine_learning_recommendations()` - Get recommendations from the machine learning model. 
* `text_pipeline()` - Convert text from the email to machine readable text by removing HTML design elements.
* `detect_language()` - Detect the language of the email. 
