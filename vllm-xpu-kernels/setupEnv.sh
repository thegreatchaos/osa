git submodule add https://github.com/vllm-project/vllm-xpu-kernels.git src
python3.12 -m venv .env
source env.sh
pip install --upgrade pip
pip install -r src/requirements.txt

