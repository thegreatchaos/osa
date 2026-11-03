source ../../env.sh
set -x
python noPerf.py fp8 2>&1 | tee ~/tmp/kvDtypeFp8_noPerf.log
python noPerf.py 2>&1 | tee ~/tmp/kvDtypeFp16_noPerf.log
