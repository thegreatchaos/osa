export ROOT="$(cd $(dirname "$1") && pwd)"
source /opt/intel/oneapi/setvars.sh
source .env/bin/activate
export PS1='>'
