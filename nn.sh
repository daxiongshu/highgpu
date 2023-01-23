pip install -r requirement.txt
python /highgpu/run.py -g 0 &
python /highgpu/run.py -g 1 &
python /highgpu/run.py -g 2 &
python /highgpu/run.py -g 3 &
