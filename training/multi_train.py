import subprocess

seed_min = 990
seed_max = 1000
filename = 'model16'
for seed in range(seed_min, seed_max):
    print(f"Running with seed {seed}")
    subprocess.run(['python3', 'training/training.py', '-f', 'model15-', '-e', '4', '-s', str(seed)])
