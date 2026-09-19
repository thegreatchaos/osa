cp dockerUniPerf.sh ~/prjs/models/perf.sh
cp dockerQwen38-27B.py ~/prjs/models/qwen38-27B.py
docker run  -it \
	    --device=/dev/dri \
	    -v /dev/dri/by-path:/dev/dri/by-path:ro \
	    -v /mnt/hdd0/models:/models \
	    scaler-dev-unitrace:latest  
