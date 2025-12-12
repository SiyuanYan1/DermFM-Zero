conda create -n PanDerm-v2-SAE python=3.10.19
conda activate PanDerm-v2-SAE
pip install -r requirements.txt
cd automated-concept-discovery
pip install -e .
cd ..