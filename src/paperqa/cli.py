"""命令行只装配参数，业务仍走同一组入口。"""

import argparse
import asyncio
import json
from pathlib import Path

from paperqa.agents.main import agent_query
from paperqa.agents.search import get_directory_index
from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.utils import dump_json, save_result


async def run(args):
    settings = Settings.from_file(args.settings)
    if args.papers:
        settings.index.paper_directory = str(args.papers)
    if args.index_name:
        settings.index.index_name = args.index_name
    index = await get_directory_index(settings, build=not args.reuse_index)
    if args.command in {"index", "inspect"}:
        for item in index.document_list():
            doc, texts = index.get_saved_object(item["doc_id"])
            print(
                dump_json(
                    {
                        "doc_id": doc.dockey,
                        "name": doc.docname,
                        "chunks": len(texts),
                        "tables": sum(t.kind == "table" for t in texts),
                        "parse_warnings": doc.parse_warnings,
                    }
                )
            )
        print(f"解析与索引完成；版本：{index.generation or 'legacy'}，本地构建：{index.build_stats}")
        return 0
    docs = Docs(settings)
    if args.preload:
        for doc_id in index.doc_ids:
            docs.aadd_texts(*index.get_saved_object(doc_id))
    spec = json.loads(args.spec.read_text(encoding="utf-8")) if args.spec else None
    print(f"问题：{args.question}\n正在运行 Agent，请等待工具处理完成……", flush=True)
    response = await agent_query(args.question, settings, docs, spec, index)
    run_path = save_result(response, args.output)
    print(response.answer)
    print(f"\n运行状态：{response.status}；工具历史：{response.tool_history}")
    print(f"用量：{dump_json(response.usage)}\n记录：{run_path.resolve()}")
    return 2 if response.status in {"error", "timeout"} else 0


def main():
    parser = argparse.ArgumentParser(description="论文实验对照与证据核验；默认本地资料库模式")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("index", "inspect", "ask"):
        p = sub.add_parser(command)
        p.add_argument("--papers", type=Path)
        p.add_argument("--settings", type=Path)
        p.add_argument("--index-name")
        p.add_argument("--reuse-index", action="store_true")
        if command == "ask":
            p.add_argument("--question", required=True)
            p.add_argument("--spec", type=Path)
            p.add_argument("--preload", action="store_true")
            p.add_argument("--output", type=Path, default=Path("runs"))
    try:
        raise SystemExit(asyncio.run(run(parser.parse_args())))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"输入或材料错误：{exc}\n")


if __name__ == "__main__":
    main()
