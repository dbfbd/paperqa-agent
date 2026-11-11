"""FastAPI 接口。问答固定快照，入库独立排队构建。"""

import asyncio
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import Field

from paperqa.agents.main import agent_query
from paperqa.agents.search import SearchIndex, get_directory_index
from paperqa.catalog import Catalog, IdempotencyConflict, PublishConflict
from paperqa.settings import Settings
from paperqa.types import ComparisonSpec, Model
from paperqa.utils import dump_json, save_result, stable_id


class QueryRequest(Model):
    question: str = Field(min_length=1, max_length=12000)
    doc_ids: list[str] | None = None
    spec: ComparisonSpec | None = None


def create_app(settings=None, output_directory="runs"):
    settings = settings or Settings.from_file(os.getenv("PQA_SETTINGS"))
    write_lock, view_lock, catalog_lock = asyncio.Lock(), asyncio.Lock(), asyncio.Lock()
    query_slots = asyncio.Semaphore(settings.agent.concurrent_queries)
    submit_lock = asyncio.Lock()
    upload_slots = asyncio.Semaphore(settings.index.pending_uploads)
    builds = {}
    catalog = current = None
    retrieval_cache = None
    if redis_url := os.getenv("PQA_REDIS_URL") or settings.index.redis_url:
        from paperqa.cache import RedisCandidateCache

        retrieval_cache = RedisCandidateCache(redis_url, settings.index.redis_ttl)

    async def get_catalog():
        nonlocal catalog
        if catalog is None:
            async with catalog_lock:
                if catalog is None:
                    catalog = await asyncio.to_thread(Catalog, settings.index)
        return catalog

    @asynccontextmanager
    async def lifespan(_app):
        database = await get_catalog()
        await asyncio.to_thread(database.interrupt_unfinished)
        try:
            yield
        finally:
            await asyncio.gather(*builds.values(), return_exceptions=True)
            if retrieval_cache:
                await retrieval_cache.close()

    app = FastAPI(title="论文实验对照与证据核验", lifespan=lifespan)

    async def get_index():
        nonlocal current
        async with view_lock:
            current = await get_directory_index(
                settings, catalog=await get_catalog(), current=current
            )
            return current

    @app.get("/")
    async def page():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    async def build(job, incoming):
        nonlocal current
        try:
            async with write_lock:
                await asyncio.to_thread(catalog.update_job, job, "building")
                root = Path(settings.index.paper_directory).resolve()
                await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
                index = await get_directory_index(
                    settings, build=True, catalog=catalog, current=current,
                    job=job, incoming=incoming,
                )
                current = index
                return {
                    "job_id": job, "index_version": index.generation,
                    "documents": index.document_list(), "build": index.build_stats,
                }
        except Exception as exc:  # noqa: BLE001 -- 后台任务的失败必须写回状态。
            code = 409 if isinstance(exc, PublishConflict) else 422
            detail = str(exc)[:400] or type(exc).__name__
            await asyncio.to_thread(catalog.update_job, job, "failed", error=detail, code=code)
            return {"error_code": code, "detail": detail}

    async def submit(database, incoming, request_key):
        # 入库路径已包含内容哈希和文件名。批次指纹与上传顺序无关。
        fingerprint = (
            stable_id(dump_json(sorted(str(path) for path in incoming))) if request_key else None
        )
        # SQLite 负责跨请求唯一性，短锁使同进程的重试能拿到同一个任务句柄。
        async with submit_lock:
            try:
                job, created = await asyncio.to_thread(
                    database.new_job, request_key, fingerprint,
                    accept_new=len(builds) < settings.index.pending_uploads,
                )
            except IdempotencyConflict as exc:
                raise HTTPException(422, str(exc)) from exc
            if job is None:
                raise HTTPException(429, "入库队列已满，请稍后重试")
            if created:
                task = asyncio.create_task(build(job, incoming))
                builds[job] = task
                task.add_done_callback(lambda _: builds.pop(job, None))
            else:
                incoming.clear()
            return job, builds.get(job), created

    @app.post("/documents")
    async def upload(
        files: Annotated[list[UploadFile], File()],
        wait: bool = True,
        request_key: Annotated[str | None, Header(
            alias="Idempotency-Key", min_length=1, max_length=128
        )] = None,
    ):
        if request_key is None and len(builds) >= settings.index.pending_uploads:
            raise HTTPException(429, "入库队列已满，请稍后重试")
        async with upload_slots:
            root = Path(settings.index.paper_directory).resolve()
            incoming, total = {}, 0
            for file in files:
                name = re.sub(r'[\\/:*?"<>|]', "_", file.filename or "document")
                if Path(name).suffix.lower() not in {".pdf", ".txt"}:
                    raise HTTPException(400, "仅支持 PDF/TXT")
                content = await file.read(64 * 1024 * 1024 + 1)
                if not content or len(content) > 64 * 1024 * 1024:
                    raise HTTPException(400, "单文件应非空且不超过 64 MiB")
                total += len(content)
                if total > 128 * 1024 * 1024:
                    raise HTTPException(413, "单次上传不超过 128 MiB")
                path = root / (stable_id(content) + "-" + name)
                if not path.resolve().is_relative_to(root):
                    raise HTTPException(400, "上传路径不属于资料目录")
                incoming[path] = content

            database = await get_catalog()
            job, task, created = await asyncio.shield(submit(database, incoming, request_key))

        if not wait and created:
            return JSONResponse({"job_id": job, "status": "queued"}, status_code=202)
        if wait and task is not None:
            result = await asyncio.shield(task)
            if "error_code" in result:
                raise HTTPException(result["error_code"], result["detail"])
            return result if created else {**result, "replayed": True}

        recorded = await asyncio.to_thread(database.job, job)
        state = recorded["status"]
        if state in {"queued", "building"} or not wait:
            return JSONResponse(
                {"job_id": job, "status": state, "index_version": recorded["version"]},
                status_code=202 if state in {"queued", "building"} else 200,
            )
        if state == "failed":
            raise HTTPException(recorded["error_code"] or 422, recorded["error"] or "入库失败")
        version = recorded["version"]
        manifest = await asyncio.to_thread(database.version_manifest, version)
        index = await asyncio.to_thread(SearchIndex, settings, database, version, manifest)
        return {
            "job_id": job, "index_version": version, "documents": index.document_list(),
            "build": index.build_stats, "replayed": True,
        }

    @app.get("/ingestions/{job_id}")
    async def ingestion(job_id: str):
        database = await get_catalog()
        job = await asyncio.to_thread(database.job, job_id)
        if job is None:
            raise HTTPException(404, "入库任务不存在")
        return job

    @app.get("/documents")
    async def documents():
        try:
            index = await get_index()
        except FileNotFoundError:
            return {"documents": [], "index_version": None}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"documents": index.document_list(), "index_version": index.generation}

    @app.post("/query")
    async def query(request: QueryRequest):
        async with query_slots:
            try:
                index = await get_index()
            except FileNotFoundError as exc:
                raise HTTPException(400, "请先上传材料或通过 CLI 建立索引") from exc
            except ValueError as exc:
                raise HTTPException(400, "索引不可用或解析配置已更新，请重建索引") from exc
            result = await agent_query(
                request.question, settings, spec=request.spec, index=index,
                allowed_doc_ids=request.doc_ids,
                retrieval_cache=retrieval_cache,
            )
            await asyncio.to_thread(save_result, result, output_directory)
            return {**result.model_dump(mode="json"), "index_version": index.generation}

    @app.get("/sources/{doc_id}")
    async def source(doc_id: str):
        database = await get_catalog()
        item = await asyncio.to_thread(database.source, doc_id)
        if item:
            path, name = item
            root = database.directory / "sources"
        else:
            try:
                index = await get_index()
                if index.generation:
                    raise ValueError("文档未发布")
                doc, _ = await asyncio.to_thread(index.get_saved_object, doc_id)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(404, "来源不存在") from exc
            path, name = Path(doc.source_path), doc.docname
            root = Path(settings.index.paper_directory).resolve()
        path = path.resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise HTTPException(404, "原始文件已移动或不在资料目录")
        return FileResponse(
            path, filename=name,
            media_type="application/pdf" if path.suffix.lower() == ".pdf" else "text/plain",
            content_disposition_type="inline",
        )

    return app


app = create_app()
