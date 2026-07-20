# 1. Create the conda environment
conda create -n snake_rl python=3.12.12 -y

# 2. Activate the environment
conda activate snake_rl

# 3. Ensure pip is up to date
python -m pip install --upgrade pip

# 4. Install the python modules from requirement file
pip install -r requirements.txt

# 5. Reproduce training
python train.py

# 6. Reproduce evaluation
python evaluate.py