import os
import json
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nest_asyncio
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_files()
    print("✅ 서버 초기화 완료")
    yield
    print("🚪 서버 종료 중")


app = FastAPI(lifespan=lifespan)
nest_asyncio.apply()


class QuestionRequest(BaseModel):
    question: str
    purpose: str
    prompt_type: Optional[str] = "default"


class TaskResponse(BaseModel):
    task_id: str
    status: str


QUEUE_FILE = "queue.json"
ANSWER_FILE = "answers.json"


def init_files():
    for file in [QUEUE_FILE, ANSWER_FILE]:
        if not Path(file).exists():
            with open(file, "w") as f:
                json.dump([], f)


@app.post("/ask", response_model=TaskResponse)
async def ask_question(request: QuestionRequest):
    task_id = str(uuid.uuid4())
    new_task = {
        "id": task_id,
        "question": request.question,
        "purpose": request.purpose,
        "prompt_type": request.prompt_type,
        "status": "pending",
    }
    try:
        with open(QUEUE_FILE, "r+") as f:
            tasks = json.load(f) if Path(QUEUE_FILE).stat().st_size > 0 else []
            tasks.append(new_task)
            f.seek(0)
            json.dump(tasks, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"큐 업데이트 실패: {str(e)}")
    return {"task_id": task_id, "status": "queued"}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    try:
        if Path(ANSWER_FILE).exists():
            with open(ANSWER_FILE, "r") as f:
                answers = json.load(f)
                for answer in answers:
                    if answer["id"] == task_id:
                        return answer
    except json.JSONDecodeError:
        pass
    if Path(QUEUE_FILE).exists():
        with open(QUEUE_FILE, "r") as f:
            tasks = json.load(f) if Path(QUEUE_FILE).stat().st_size > 0 else []
            if any(task["id"] == task_id for task in tasks):
                return {"status": "processing"}
    raise HTTPException(status_code=404, detail="작업을 찾을 수 없음")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
