export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/vtune/vtune-vars.sh --force
export PS1='\033[32mtla>\033[0m'
