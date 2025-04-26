import json
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()


async def ask_ai(
        prompt: str, 
        system_prompt: str, 
        model: str = os.getenv("OPENAI_MODEL"), 
        max_tokens: int = 50000, 
        base_url: str = os.getenv("OPENAI_BASE_URL"), 
        show_time: bool = False, 
        num_ctx: int | None = None, 
        timeout: float = 600.0
    ):
    print("Идет запрос к LLM")
    start_time = time.time()
    url = base_url + "/chat/completions"
    
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    if system_prompt:
        messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })


    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }
    
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + os.getenv("OPENAI_KEY")
    }

    # Изменение на использование httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=timeout)

    chat_result = response.json()
    answer = chat_result['choices'][0]['message']['content']
    tokens_answer = chat_result['usage']['completion_tokens']
    model_answer = chat_result['model']
    
    if show_time:
        end_time = time.time()
        print(f"Время выполнения: {end_time - start_time} секунд")

    return answer, tokens_answer, model_answer

