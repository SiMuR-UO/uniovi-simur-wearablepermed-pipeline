# Description
Uniovi Simur WearablePerMed Pipeline

# Execute
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt

python3 main.py \
    --verbose \
    --execute-steps 2,3,4,5 \
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