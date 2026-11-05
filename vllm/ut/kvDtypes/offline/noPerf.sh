source ../../env.sh
set -x
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python noPerf.py fp8 2>&1 | tee ~/tmp/kvDtypeFp8_noPerf.log
python noPerf.py 2>&1 | tee ~/tmp/kvDtypeFp16_noPerf.log
