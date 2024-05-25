# These are the default commands included in the original repository
python generate_dataset.py --initial_vel 1
python generate_dataset.py --initial_vel 0 --length 2000 --length_test 2000
python generate_dataset.py --n_balls 15 --initial_vel 1

# The one we use (based on the details provided in the paper):
python generate_dataset.py --initial_vel 1 --num-train 3000 --length 1000 --length_test 1000 --sufix "small"
