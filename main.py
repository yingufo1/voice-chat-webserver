from typing import Union

from fastapi import FastAPI, UploadFile
import whisper
import requests
import pyttsx3

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

model = whisper.load_model("base")  # 根据性能选择base/small/medium

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    # 保存上传的音频
    with open("temp_audio.mp3", "wb") as f:
        f.write(await file.read())
    
    # Whisper转文本
    result = model.transcribe("temp_audio.mp3")
    text = result["text"]
    
    # 发送到LLaMA服务器
    llama_response = "i am a placeholder for llama response"
    #  requests.post(
    #     "http://your-gpu-server:5000/api/v1/generate",
    #     json={"prompt": text, "max_length": 200}
    # ) 
    
    # 文本转语音
    engine = pyttsx3.init()
    engine.save_to_file(llama_response.json()["results"][0]["text"], "response.mp3")
    engine.runAndWait()
    
    return {"response": "response.mp3"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0", port=8000)