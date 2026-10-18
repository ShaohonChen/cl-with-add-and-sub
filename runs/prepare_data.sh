python data/generate_dataset.py --a-min 0 --b-min 0 --a-max 1000 --b-max 1000 \
    --a-places 0 --b-places 0 --c-places 0 --operators "+-" \
    --count 55500 --output data/add-sub-int.json

python data/split_dataset.py data/add-sub-int.json \
  --eval-count 500 \
  --test-count 5000
