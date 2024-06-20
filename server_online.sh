# Exit if OPENAI_API_KEY is not set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Please set the environment variable OPENAI_API_KEY this way:"
  echo "export OPENAI_API_KEY=YOUR_API_KEY"
  exit 1
fi
python3 whisper_online_server.py --backend openai-api --model large-v3 --language ru --warmup-file output.mp3
