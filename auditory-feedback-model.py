from openai import OpenAI
import ollama
import pyttsx3
from pathlib import Path
import time
import os
import random

# source venv/bin/activate
# pip install -r requirements.txt

def load_api_key():
    with open('openai_api_key.txt', 'r') as file:
        return file.read()

LLM_MODEL = "gpt-4o-mini"
LOCAL_LLM_MODEL = "llama3.2"
SPEECH_MODEL = "tts-1"
SPEECH_VOICE = "alloy"
OPENAI_API_KEY = load_api_key()
client = OpenAI(api_key=OPENAI_API_KEY)

if not os.path.exists("output"):
    os.makedirs("output")

def ask_gpt(prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=LLM_MODEL,
    )
    return response.choices[0].message.content

def ask_ollama(prompt):
    response = ollama.chat(model=LOCAL_LLM_MODEL, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    load_time = response['load_duration'] / 1e9
    total_time = response['total_duration'] / 1e9
    print(f"loaded\t\t{load_time:.3f}s")
    return response['message']['content']

def speak_text_whisper(text):
    speech_file_path = Path(__file__).parent / "output/speech-whisper.mp3"
    response = client.audio.speech.create(
        model=SPEECH_MODEL,
        voice=SPEECH_VOICE,
        input=text
    )
    with open(speech_file_path, "wb") as file:
        file.write(response.content)

def speak_text_local(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 190)
    engine.save_to_file(text, "output/speech-local.aiff")
    engine.runAndWait()

def load_reasoning_file(filename):
    with open(f'WOMD-Reasoning/training/{filename}', 'r') as file:
        return file.read()

def get_random_file():
    folder_path = os.path.join(os.path.dirname(__file__), 'WOMD-Reasoning/training')
    file_list = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    if file_list:
        random_file = random.choice(file_list)
        return random_file
    else:
        return "error"

def run_model(reasoning_filename, local):
    start = time.time()

    data = load_reasoning_file("scid_1ae21c08cfa0969b__aid_465__atype_1.json")

    prompt = data + """\n\nThis is a scenario of a self-driving car. You are the car. Your job is to describe why the car is doing what it is doing to the user inside the car who is not driving. Your output will be spoken aloud to the user so keep it brief. Do not use any technical terms like "ego agent" just call the other agents cars. Do not mention any specific numbers, just mention if a car is going fast or slow relative to yourself if it is important. Only mention important information and keep it mainly about yourself. Write ONE BRIEF sentence describing why you are doing what you are doing right now."""

    start_inference = time.time()
    if local:
        response = ask_ollama(prompt)
    else:
        response = ask_gpt(prompt)

    print(response)
    end_inference = time.time()
    print(f"inference\t{end_inference - start_inference:.3f}s")

    start_tts = time.time()
    if local:
        speak_text_local(response)
    else:
        speak_text_whisper(response)
    
    end_tts = time.time()
    print(f"tts\t\t{end_tts - start_tts:.3f}s")
    print(f"total\t\t{end_tts - start:.3f}s")
    return (end_inference - start_inference, end_tts - start_tts, end_tts - start)

if __name__ == "__main__":
    filename = get_random_file()
    # print(filename)
    # print("==== local ====")
    run_model(filename, True)
    # print("\n==== online ====")
    # run_model(filename, False)

    inf_times = []
    tts_times = []
    total_times = []
    for _ in range(20):
        filename = get_random_file()
        print(filename)
        output = run_model(filename, True)
        inf_times.append(output[0])
        tts_times.append(output[1])
        total_times.append(output[2])
    print(inf_times)
    print(tts_times)
    print(total_times)
    inf_avg = sum(inf_times) / len(inf_times)
    tts_avg = sum(tts_times) / len(tts_times)
    total_avg = sum(total_times) / len(total_times)
    print(f"inference avg\t{inf_avg:.3f}s")
    print(f"tts avg\t\t{tts_avg:.3f}s")
    print(f"total avg\t{total_avg:.3f}s")
