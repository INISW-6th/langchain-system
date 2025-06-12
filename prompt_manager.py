import json
from pathlib import Path


class PromptManager:
    def __init__(self, prompt_file: str = "prompt.json"):
        """
        :param prompt_file: 기능/목적별 프롬프트가 저장된 JSON 파일 경로
        """
        self.prompt_file = Path(prompt_file)
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_prompt(self, key: str, prompt_type: str = "default") -> str:
        """
        기능명 또는 목적명(key)과 타입(prompt_type)으로 프롬프트 템플릿 반환
        :param key: 기능명 또는 목적명(예: 'qa', 'summarize', 'medical')
        :param prompt_type: 타입(예: 'default', 'detailed', 'cot' 등)
        """
        if key not in self.prompts:
            raise ValueError(f"'{key}'에 해당하는 프롬프트가 없습니다.")
        if isinstance(self.prompts[key], dict):
            # 타입별 프롬프트 관리
            if prompt_type not in self.prompts[key]:
                raise ValueError(f"'{key}'에 '{prompt_type}' 타입 프롬프트가 없습니다.")
            return self.prompts[key][prompt_type]
        # 단일 템플릿(문자열) 지원
        return self.prompts[key]

    def format_prompt(self, key: str, prompt_type: str = "default", **kwargs) -> str:
        """
        프롬프트에 변수값을 채워 완성된 프롬프트 반환
        :param kwargs: {context=..., question=..., ...}
        """
        template = self.get_prompt(key, prompt_type)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"프롬프트에 필요한 변수 누락: {e}")
