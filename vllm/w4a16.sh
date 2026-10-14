source env.sh
cd kerns
python -m pytest tests/test_int4_gemm_w4a16_trace.py -v -s
