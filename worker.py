import os
import json
import time
import torch
import gc
from pathlib import Path
from ModularRAGExperiment import ModularRAGExperiment
from config import experiment_config, docs
from prompt_manager import PromptManager

QUEUE_FILE = "queue.json"
ANSWER_FILE = "answers.json"


def clear_gpu_memory():
    """GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_tasks():
    if not Path(QUEUE_FILE).exists():
        return []
    try:
        with open(QUEUE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_tasks(tasks):
    with open(QUEUE_FILE, "w") as f:
        json.dump(tasks, f, indent=2)


def append_answer(answer):
    answers = []
    if Path(ANSWER_FILE).exists():
        try:
            with open(ANSWER_FILE, "r") as f:
                answers = json.load(f)
        except json.JSONDecodeError:
            pass
    answers.append(answer)
    with open(ANSWER_FILE, "w") as f:
        json.dump(answers, f, indent=2)


def main():
    rag = ModularRAGExperiment(experiment_config, docs)
    prompt_manager = PromptManager("prompt.json")
    print("✅ 워커 시작 (Ctrl+C로 중단)")

    while True:
        tasks = load_tasks()
        pending_tasks = [task for task in tasks if task["status"] == "pending"]

        if not pending_tasks:
            time.sleep(3)
            continue

        for task in pending_tasks:
            try:
                # 프롬프트 템플릿 선택
                prompt_template = prompt_manager.get_prompt(
                    task["purpose"], task.get("prompt_type", "default")
                )
                # 답변 생성
                answer_text = rag.ask_modular_rag(
                    purpose=task["purpose"],
                    question=task["question"],
                    prompt_template=prompt_template,
                )
                # 결과 기록
                result = {
                    "id": task["id"],
                    "question": task["question"],
                    "answer": answer_text,
                    "status": "completed",
                }
                append_answer(result)
                # 큐 상태 갱신
                for t in tasks:
                    if t["id"] == task["id"]:
                        t["status"] = "completed"
                save_tasks(tasks)
                print(f"✅ 처리 완료: {task['id']}")
            except Exception as e:
                print(f"❌ 처리 실패: {task['id']} - {e}")
            finally:
                clear_gpu_memory()
        time.sleep(2)


if __name__ == "__main__":
    main()
