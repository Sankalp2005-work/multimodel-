import os
import whisper
import sys

def main():
    audio_dir = r"c:\Users\sankalp\Desktop\IIT Mandi\Audios"
    if not os.path.exists(audio_dir):
        print(f"Directory {audio_dir} does not exist.")
        return

    print("Loading whisper model (base)...")
    model = whisper.load_model("base")

    for filename in os.listdir(audio_dir):
        if filename.endswith(".mp4") or filename.endswith(".m4a") or filename.endswith(".wav"):
            path = os.path.join(audio_dir, filename)
            print(f"Transcribing {filename}...")
            try:
                result = model.transcribe(path)
                print(f"[{filename}] Transcribed Text: {result['text']}\n")
            except Exception as e:
                print(f"[{filename}] Failed to transcribe: {e}\n")

if __name__ == "__main__":
    main()
