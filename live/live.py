import sounddevice as sd
from whisper_live.client import TranscriptionClient

def list_audio_devices():
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        input_channels = device['max_input_channels']
        output_channels = device['max_output_channels']
        device_type = "Input/Output" if input_channels > 0 and output_channels > 0 else "Input" if input_channels > 0 else "Output"
        print(f"{i}: {device['name']} ({device_type})")
        print(f"   Input channels: {input_channels}, Output channels: {output_channels}")
        print(f"   Sample rates: {device['default_samplerate']}")
        print(f"   Device type: {device['hostapi']}")
        print()

def select_audio_device():
    list_audio_devices()
    while True:
        try:
            device_id = int(input("Enter the number of the device you want to use: "))
            device_info = sd.query_devices(device_id)
            if device_info['max_input_channels'] > 0:
                return device_id
            else:
                print("This device doesn't support audio input. Please choose another.")
        except (ValueError, sd.PortAudioError):
            print("Invalid input. Please enter a valid device number.")

# List the audio devices
list_audio_devices()

# Uncomment the following lines when you're ready to select a device and proceed with transcription
selected_device = select_audio_device()

client = TranscriptionClient(
    "localhost",
    9090,
    lang="en",
    translate=False,
    model="large-v3",
    use_vad=True,
)

# Use the selected device
with sd.InputStream(device=selected_device):
    client()

print('Done')