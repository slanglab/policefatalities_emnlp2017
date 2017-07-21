mkdir -p test_batches/
rm test_batches/*
mkdir -p test_processed/
rm test_processes/*
VECTOR_LOCATION='../../pretrained/GoogleNews-vectors-negative300.bin'

cat train.json test.json > data/all.json
python preproc_A.py 'data/all.json'
# this is a minor modification of Yoon Kims's preprocessing script
python preproc_B.py $VECTOR_LOCATION
./send_to_gpu.sh
