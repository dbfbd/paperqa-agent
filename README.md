# PaperQA

基于 [Future-House/paper-qa](https://github.com/Future-House/paper-qa) 二次开发的本地论文问答与实验对照工具。通过单 Agent 工具循环查找论文、提炼带条件的事实并回查原文，输出附页码和来源的回答。

- 多语 E5 与 Tantivy BM25 混合检索、RRF 融合和按需重排。
- 按对象与指标检查证据覆盖，支持定向补查及实验条件核对。
- FastAPI 提供后台入库、任务状态查询、问答与原文下载。问答固定索引版本，新索引完成后原子发布。
- 复用解析结果和持久化向量，支持可选 Redis 候选缓存、超时熔断与本地检索降级。

## 安装和启动

需要 Python 3.11 或以上版本及 uv，项目固定使用 Python 3.13。以下命令在仓库根目录运行，环境变量示例使用 PowerShell。

```powershell
uv sync --locked --extra rag --extra redis
$env:DEEPSEEK_API_KEY = 'your-key'
$env:PQA_SETTINGS = 'configs/rag-local.json'
uv run --locked --extra rag --extra redis uvicorn app.api:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 上传 PDF/TXT 并提问，接口文档位于 `/docs`。首次使用会下载本地 E5 模型，触发重排时会加载 CrossEncoder。处理期间可查询入库任务状态，重复上传可携带 `Idempotency-Key` 复用任务。

可选配置 `PQA_REDIS_URL`，例如 `redis://127.0.0.1:6379/0`。未配置 Redis 时使用本地检索。

## 命令行

将文本型 PDF 或 UTF-8 TXT 放入 `papers` 目录，再执行索引和问答。

```powershell
uv run --locked --extra rag pqa index --settings configs/rag-local.json
uv run --locked --extra rag pqa ask --settings configs/rag-local.json --reuse-index --question '比较论文中各模型的实验条件和结果'
```

索引默认保存在 `.paperqa-v1`，问答结果保存在 `runs`。配置入口为 `configs/rag-local.json`，可调整模型、资料目录、检索与预算设置。

## 使用范围与许可证

面向本地单服务进程，支持文本型 PDF/TXT。表格列关系和实验条件仍需结合原文核对。

本项目在上游基础上重构了检索、证据处理和服务接口。遵循 [Apache-2.0](LICENSE)，保留上游版权声明。
