# curl -sO https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.yaml
# docker compose up --detach faster-whisper-server-cuda
# Make dir for cache if not exists
mkdir -p cache
docker run \
    -it \
    --gpus all \
    -v $(pwd)/cache:/root/.cache/huggingface \
    -p 9090:9090 \
    ghcr.io/collabora/whisperlive-gpu:latest