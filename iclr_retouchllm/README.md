# RetouchLLM

# 1. Create conda environment
conda env create -f environment.yml -n retouchllm
conda activate retouchllm

# 2. Download pretrained model
# Download from https://huggingface.co/OpenGVLab/InternVL3-14B-hf

# 3. Download Reference images and gt image
python img_download.py

# 4. Run
python retouchllm.py --root_dir ./iclr_retouchllm
