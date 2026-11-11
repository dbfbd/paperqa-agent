"""场景规则使用普通函数；不请求模型、不读取磁盘、不持有会话副本。"""

from decimal import Decimal, InvalidOperation

from paperqa.utils import dump_json, normalized

CONDITION_KEYS = ("dataset", "split", "metric", "extra_training_data")


def _numeric(value):
    try:
        number = Decimal(value.strip())
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def _conditions(fact):
    return tuple((k, normalized(v or "")) for k, v in sorted(fact.conditions.items()))


def _same_experiment(a, b):
    # 完整条件可跨片段合并；未知条件只允许同一原文摘录去重，不能作通配。
    complete = all(a.conditions.get(k) and b.conditions.get(k) for k in CONDITION_KEYS)
    if complete:
        return _conditions(a) == _conditions(b)
    return (
        _conditions(a) == _conditions(b)
        and bool(a.sources)
        and bool(b.sources)
        and {(s.chunk_id, normalized(s.quote)) for s in a.sources}
        == {(s.chunk_id, normalized(s.quote)) for s in b.sources}
    )


def merge_facts(existing, candidates):
    """返回新事实池与变化标记；不会原地修改输入。"""
    merged = [f.model_copy(deep=True) for f in existing]
    changed = False
    for incoming in candidates:
        same = [
            f
            for f in merged
            if (f.object_id, f.dimension) == (incoming.object_id, incoming.dimension)
            and f.status != "superseded"
            and _same_experiment(f, incoming)
        ]
        duplicate = next(
            (
                f
                for f in same
                if normalized(f.value) == normalized(incoming.value)
                and f.unit == incoming.unit
                and (f.status == "unsupported") == (incoming.status == "unsupported")
            ),
            None,
        )
        if duplicate:
            known = {(s.chunk_id, s.quote) for s in duplicate.sources}
            additions = [s for s in incoming.sources if (s.chunk_id, s.quote) not in known]
            duplicate.sources.extend(additions)
            changed = changed or bool(additions)
            continue
        fact = incoming.model_copy(deep=True)
        merged.append(fact)
        changed = True
    # 冲突由当前仍有效的候选推导，避免旧事实被替代后残留冲突标签。
    active = [f for f in merged if f.status in {"candidate", "conflict"}]
    for fact in active:
        value = _numeric(fact.value)
        conflict = (
            value is not None
            and all(fact.conditions.get(k) for k in CONDITION_KEYS)
            and any(
                other is not fact
                and (other.object_id, other.dimension) == (fact.object_id, fact.dimension)
                and other.unit == fact.unit
                and _same_experiment(fact, other)
                and _numeric(other.value) not in (None, value)
                for other in active
            )
        )
        status = "conflict" if conflict else "candidate"
        if fact.status != status:
            fact.status = status
            changed = True
    return merged, changed


def compute_coverage(spec, facts):
    rows = []
    for obj in spec.objects:
        for dim in spec.dimensions:
            matches = [
                f
                for f in facts
                if f.object_id == obj.object_id
                and f.dimension == dim
                and f.sources
                and f.status not in {"unsupported", "superseded"}
            ]
            state = (
                "missing"
                if not matches
                else ("conflict" if any(f.status == "conflict" for f in matches) else "found")
            )
            rows.append(
                {
                    "object_id": obj.object_id,
                    "dimension": dim,
                    "status": state,
                    "fact_ids": [f.fact_id for f in matches],
                }
            )
    return rows


def check_comparability(spec, facts):
    """只判断明确字段差异；不同表述不自动构成语义矛盾。"""
    result = []
    for key in CONDITION_KEYS:
        per_object = {}
        for obj in spec.objects:
            values = {
                f.conditions[key]
                for f in facts
                if f.object_id == obj.object_id
                and f.status not in {"unsupported", "superseded"}
                and f.conditions.get(key)
            }
            per_object[obj.object_id] = sorted(values)
        if any(not values for values in per_object.values()):
            result.append({"condition": key, "status": "unknown", "values": per_object})
        elif len({tuple(values) for values in per_object.values()}) > 1:
            result.append(
                {
                    "condition": key,
                    "status": "different",
                    "values": per_object,
                    "note": "原文条件字段不同；需解释其可比性，不能据此直接归因模型优劣",
                }
            )
    return result


def select_contexts(spec, facts, contexts, reads, budget, max_sources, question=""):
    supported = [
        f for f in facts if f.sources and f.status not in {"unsupported", "superseded"}
    ]
    score = {c.chunk_id: c.score for c in contexts}
    ranked = sorted(supported, key=lambda f: -max(score.get(s.chunk_id, 0) for s in f.sources))
    selected, raw_reads, source_ids = [], [], set()
    cells = [(o.object_id, d) for o in spec.objects for d in spec.dimensions]
    # 先轮流照顾所有对象/维度，再加剩余事实；冲突同组作为一个不可拆的单元。
    groups, rest, scheduled = [], [], set()
    for obj, dim in cells:
        matches = [f for f in ranked if (f.object_id, f.dimension) == (obj, dim)]
        cell_groups = []
        for first in matches:
            if first.fact_id in scheduled:
                continue
            group = [
                f
                for f in matches
                if first.status == "conflict"
                and f.status == "conflict"
                and _same_experiment(first, f)
            ] or [first]
            scheduled.update(f.fact_id for f in group)
            cell_groups.append(group)
        if cell_groups:
            groups.append(cell_groups[0])
            rest.extend(cell_groups[1:])

    pool = compute_coverage(spec, facts)

    def snapshot(chosen, chosen_reads):
        actual = compute_coverage(spec, chosen)
        # 区分“没有取得依据”与“已有依据但未入选回答”，后者不能被写成论文未报告。
        for p, a in zip(pool, actual):
            if p["status"] != "missing" and a["status"] == "missing":
                a["status"] = "excluded_by_budget"
        ids = {f.fact_id for f in chosen}
        read_ids = {r.read_id for r in chosen_reads}
        return {
            "facts": chosen,
            "source_reads": chosen_reads,
            "coverage": actual,
            "pool_coverage": pool,
            "excluded_fact_ids": [f.fact_id for f in supported if f.fact_id not in ids],
            "excluded_source_read_ids": [r.read_id for r in reads if r.read_id not in read_ids],
            "source_read_issues": [
                {
                    "doc_id": r.doc_id,
                    "pages": r.pages,
                    "issue": "回查内容被截短或版面关系未恢复，未用作回答原文",
                }
                for r in reads
                if r.layout_uncertain or r.truncated
            ],
        }

    def size(chosen, chosen_reads):
        return len(
            dump_json(build_answer_context(spec, snapshot(chosen, chosen_reads), question))
        )

    def add_group(group):
        nonlocal source_ids
        ids = {s.chunk_id for f in group for s in f.sources}
        if size(selected + group, raw_reads) > budget or len(source_ids | ids) > max_sources:
            return
        selected.extend(group)
        source_ids |= ids

    for group in groups:
        add_group(group)
    # 回查只读取相关文档或明确目标，不把历史所有原文无条件塞给回答模型。
    allowed = set(spec.allowed_doc_ids)
    target_ids = {o.object_id for o in spec.objects}
    for read in reversed(reads):
        if read.layout_uncertain or read.truncated:
            continue
        if allowed and read.doc_id not in allowed:
            continue
        if read.targets and not target_ids.intersection(read.targets):
            continue
        if (
            size(selected, raw_reads + [read]) <= budget
            and len(source_ids | set(read.chunk_ids)) <= max_sources
        ):
            raw_reads.append(read)
            source_ids.update(read.chunk_ids)
    for group in rest:
        add_group(group)
    result = snapshot(selected, raw_reads)
    result["used_chars"] = size(selected, raw_reads)
    result["budget_too_small"] = result["used_chars"] > budget
    return result


def build_answer_context(spec, selection, question):
    facts = selection["facts"]
    rows = []
    for obj in spec.objects:
        cells = {}
        for dim in spec.dimensions:
            matched = [f for f in facts if f.object_id == obj.object_id and f.dimension == dim]
            coverage = next(
                c
                for c in selection["coverage"]
                if (c["object_id"], c["dimension"]) == (obj.object_id, dim)
            )
            cells[dim] = {
                "status": coverage["status"],
                "fact_ids": [f.fact_id for f in matched],
            }
        rows.append({"object_id": obj.object_id, "name": obj.name, "cells": cells})
    data = {
        "question": question,
        "mode": spec.mode,
        "dimensions": spec.dimensions,
        "rows": rows,
        "facts": [f.model_dump() for f in facts],
        "source_reads": [r.model_dump() for r in selection["source_reads"]],
        "coverage": selection["coverage"],
        "pool_coverage": selection["pool_coverage"],
        "excluded_fact_ids": selection["excluded_fact_ids"],
        "excluded_source_read_ids": selection.get("excluded_source_read_ids", []),
        "source_read_issues": selection.get("source_read_issues", []),
        "comparability": check_comparability(spec, facts),
    }
    return data


def summarize_progress(spec, before, after):
    return {
        "new_fact_count": len(after) - len(before),
        "coverage": compute_coverage(spec, after),
        "unsupported_fact_ids": [f.fact_id for f in after if f.status == "unsupported"],
    }


def render_table(data):
    """Markdown 与页面共用同一份单元格数据，数值不再经模型重写。"""

    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    dimensions = data["dimensions"]
    lookup = {f["fact_id"]: f for f in data["facts"]}
    lines = [
        "| 实验对象 | " + " | ".join(map(escape, dimensions)) + " |",
        "|---|" + "---|" * len(dimensions),
    ]
    labels = {"missing": "尚未取得依据", "excluded_by_budget": "未进入本次回答"}
    for row in data["rows"]:
        cells = []
        for dim in dimensions:
            cell = row["cells"][dim]
            values = []
            for fact_id in cell["fact_ids"]:
                f = lookup[fact_id]
                conditions = "; ".join(f"{k}={v}" for k, v in f["conditions"].items() if v)
                values.append(
                    escape(f["value"] + f["unit"])
                    + f" [{f['fact_id']}]"
                    + (f" ({escape(conditions)})" if conditions else "")
                )
            prefix = "存在冲突：" if cell["status"] == "conflict" else ""
            cells.append(
                prefix + "; ".join(values) if values else labels.get(cell["status"], "未解决")
            )
        lines.append("| " + escape(row["name"]) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_sources(data):
    lines = []
    for f in data["facts"]:
        for source in f["sources"]:
            lines.append(
                f"- [{f['fact_id']}] 文档 {source['doc_id']}，片段 {source['chunk_id']}，"
                f"PDF页 {source['pages']}：{source['quote']}"
            )
    for read in data.get("source_reads", []):
        lines.append(
            "- 回查片段 "
            + ", ".join(f"[{c}]" for c in read["chunk_ids"])
            + f"：文档 {read['doc_id']}，PDF页 {read['pages']}（完整原文保存在运行记录）"
        )
    return "\n".join(lines)
