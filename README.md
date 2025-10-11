<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/uniovi-simur-wearablepermed-pipeline.svg?branch=main)](https://cirrus-ci.com/github/<USER>/uniovi-simur-wearablepermed-pipeline)
[![ReadTheDocs](https://readthedocs.org/projects/uniovi-simur-wearablepermed-pipeline/badge/?version=latest)](https://uniovi-simur-wearablepermed-pipeline.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/uniovi-simur-wearablepermed-pipeline/main.svg)](https://coveralls.io/r/<USER>/uniovi-simur-wearablepermed-pipeline)
[![PyPI-Server](https://img.shields.io/pypi/v/uniovi-simur-wearablepermed-pipeline.svg)](https://pypi.org/project/uniovi-simur-wearablepermed-pipeline/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/uniovi-simur-wearablepermed-pipeline.svg)](https://anaconda.org/conda-forge/uniovi-simur-wearablepermed-pipeline)
[![Monthly Downloads](https://pepy.tech/badge/uniovi-simur-wearablepermed-pipeline/month)](https://pepy.tech/project/uniovi-simur-wearablepermed-pipeline)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/uniovi-simur-wearablepermed-pipeline)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# Description

> Uniovi Simur WearablePerMed Pipeline.

## Schaffolding
Execute PyScaffold command to create the project:
```
$ putup --markdown uniovi-simur-wearablepermed-pipeline -p wearablepermed_pipeline \
     -d "Uniovi Simur WearablePerMed Pipeline." \
     -u https://github.com/SiMuR-UO/uniovi-simur-wearablepermed-pipeline.git 
```

Create a virtual environment inside for your project and active it:
```
$ python3 -m venv .venv
$ source .venv/bin/activate
```

Install and upgrade tox automation project manager:
```
$ pip install --upgrade tox
```

Install and upgrade project modules:
```
$ pip install -U pandas uniovi-simur-wearablepermed-utils
```

## Code and Debugging

Install library modules:
```
$ pip install -r requirements.txt
```

Install module locally for debugg
```
$ pip install -e .
```

Update project requirements:
```
$ pip freeze > requirements.txt
```

If you want upgrade the pipeline library to the last **uniovi-simur-wearablepermed-utils** you must execute this command (two times):

```
$ pip install uniovi-simur-wearablepermed-utils --upgrade
```

## Project management

Project commands for: test, clean, build, generate documentation or publish your library in pypi repository.
Don't forget update the version library from **setup.cfg** project build file

```
$ tox
$ tox -e clean
$ tox -e build
$ tox -e docs
$ tox -e publish -- --repository pypi
```

## Testing

Execute one pipeline from this command:
```
python3 main.py \
    --verbose \
    --execute-steps 2,3,4,5,6 \
    --dataset-folder /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input \
    --participants-missing-file /home/miguel/temp/wearablepermed_pipeline/missing_end_datetimes.csv \
    --crop-columns 1:7 \
    --window-size 250 \
    --window-overlapping-percent 50 \
    --ml-models ESANN,RandomForest \
    --ml-sensors thigh,hip,wrist \
    --output-case-folder /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/output \
    --case-id case_sample
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
