export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PATH=${PATH}:${ROOT}/cmSDK/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROOT}/cmSDK/lib
export PS1='\033[32mcm>\033[0m'
