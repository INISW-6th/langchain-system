import os
import glob
import json
import re
from typing import Dict, List
from langchain.schema import Document


def load_purpose_docs(data_path: str) -> Dict[str, List[Document]]:
    """
    목적(purpose)별로 JSON 문서를 로드하고 LangChain Document로 변환

    :param data_path: JSON 문서가 저장된 디렉토리 경로
    :return: {purpose: [Document]} 형식의 딕셔너리
    """
    json_files = glob.glob(f"{data_path}/*.json")
    purpose_docs = {}

    for file_path in json_files:
        filename = os.path.basename(file_path)
        # 파일명에서 목적 추출
        match = re.match(r"^([^_]+)_", filename)
        purpose = match.group(1) if match else os.path.splitext(filename)[0]

        with open(file_path, "r", encoding="utf-8-sig") as f:
            raw_data = json.load(f)

        if purpose not in purpose_docs:
            purpose_docs[purpose] = []

        # 문서 변환 (content와 metadata 분리)
        purpose_docs[purpose].extend(
            [
                Document(
                    page_content=item["content"],
                    metadata={**item["metadata"], "source_file": filename},
                )
                for item in raw_data
            ]
        )

    return purpose_docs


# 실험 설정 (Colab 최적화)
experiment_config = {
    # 청킹 설정
    "chunking": {
        "method": "recursive",  # recursive | fixed | custom
        "chunk_size": 700,
        "chunk_overlap": 140,
        "separators": ["\n\n", "\n", " ", ""],
    },
    # 임베딩 설정
    "embedding": {
        "model_type": "huggingface",
        "model_name": "jhgan/ko-sroberta-multitask",  # 한국어 최적화 모델
    },
    # 벡터DB 설정
    "vector_db": "faiss",  # faiss | chroma
    "initial_top_k": 20,  # 초기 검색 문서 수
    # 리랭커 설정
    "reranker": "bge",  # bge | cohere | None
    "rerank_top_k": 5,  # 리랭킹 후 최종 문서 수
    "cohere_api_key": "",  # cohere 사용 시 필요
    # LLM 설정
    "llm": "exaone",  # exaone | qwen | hyperclova | kanana
    "max_total_docs": 10,  # 다중 목적 시 최대 통합 문서 수
    # 프롬프트 템플릿 (기본값)
    "prompt_template": """
    [시스템] 다음 문서를 참고해 질문에 답변하세요:
    {context}
    
    [사용자] {question}
    [어시스턴트] 답변 (한국어 간결하게):
    """,
}

# Google Colab 경로 설정 (수정 필요)
data_path = "/content/drive/MyDrive/Textbook-Data"  # 실제 데이터 경로로 변경
purpose_docs = load_purpose_docs(data_path)
