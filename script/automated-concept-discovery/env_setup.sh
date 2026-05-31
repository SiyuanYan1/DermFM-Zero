conda create -n dermfm-zero-sae python=3.10.19
conda activate dermfm-zero-sae
pip install -r requirements.txt
cd automated-concept-discovery
pip install -e .
cd ..