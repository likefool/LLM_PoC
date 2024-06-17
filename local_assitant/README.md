# Personal Assistant with WebKnowledge


## Acknowledgement

Thanks DUY HUYNH. The voice processing & console UI code is copied from DUY's work [Build your own voice assistant and run it locally: Whisper + Ollama + Bark](https://blog.duy.dev/build-your-own-voice-assistant-and-run-it-locally/)

## Dependency

```shell:
#!pip install --upgrade --quiet langchain-nvidia-ai-endpoints
#!pip install langchain==0.1.13 langchain-community==0.0.31 langchain-core==0.1.38
#!pip install beautifulsoup4==4.12.3 numpy==1.26.4 sounddevice==0.4.6 openai-whisper==20231117 rich==13.7.1 
```

### Prerequest resources
* Nvidia NIMs API access credits, ref [Nvidia API Site](https://docs.api.nvidia.com/)
* optional: MacBook (for text-to-speech app *Say*)

## Tutorial Jupyter Notebook

Step-by-Step explains in nvidia_assistant.ipynb


## How to run the app

```shell:
python app_nvidia.py
```
