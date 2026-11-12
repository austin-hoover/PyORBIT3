#!/bin/bash

set -x 

python test_env_2d_fodo.py --sc 0
python test_env_2d_fodo.py --sc 1
python test_env_2d_fodo.py --sc 1 --offset-x 0.001
python test_env_2d_fodo.py --sc 1 --tilt 45.0
python test_env_2d_fodo_speed.py --sc 0
python test_env_2d_fodo_speed.py --sc 1
python test_env_3d_drift.py --sc 0
python test_env_3d_drift.py --sc 1
python test_env_3d_drift.py --sc 1 --rms-y 0.002 --rot-z 45.0
python test_env_3d_drift.py --sc 1 --rms-z 0.002 --rot-x 45.0

python test_env_2d_matching.py
python test_env_solenoid.py
python test_env_dispersion.py
python test_env_rf_acceleration.py
python test_env_convergence.py
python test_env_3d_focusing.py

cd sns_linac
python test_sns_linac.py --sc 0
python test_sns_linac.py --sc 1 --dist kv
python test_sns_linac.py --sc 1 --dist waterbag
python test_sns_linac.py --sc 1 --dist gauss
cd ..

cd sns_ring
python test_sns_ring.py --sc 0
python test_sns_ring.py --sc 1
python test_sns_ring.py --sc 1 --tilt 45.0
python test_sns_ring_speed.py --sc 0
python test_sns_ring_speed.py --sc 1
cd ..
