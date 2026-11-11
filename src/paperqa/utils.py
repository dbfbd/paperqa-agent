"""保留解析器所需接口，以及来源、序列化和结果保存辅助。"""

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


class ImpossibleParsingError(ValueError):
    pass


def clean_invalid_unicode(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8").replace("\x00", "")


def stable_id(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()[:24]


def normalized(text: str) -> str:
    """仅忽略排版空白，不删标点、数字或任意单词。"""
    return re.sub(r"\s+", "", text)


def dump_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.close()
            os.replace(temporary, path)
        finally:
            tmp.close()
            temporary.unlink(missing_ok=True)


def save_result(response, directory: str | Path) -> Path:
    """CLI/API 统一写出结果、可读答案与完整工具轨迹。"""
    run = Path(directory) / response.run_id
    run.mkdir(parents=True, exist_ok=True)
    payload = response.model_dump(mode="json")
    (run / "result.json").write_text(dump_json(payload), encoding="utf-8")
    (run / "answer.md").write_text(response.answer, encoding="utf-8")
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False, default=str) + "\n" for e in response.trace),
        encoding="utf-8",
    )
    return run
