# Local Image Search — agent.md / canonical project log

**Rule for every agent, including a new ChatGPT/Codex conversation:** Before edits, read agent.md and README.md, inspect main HEAD and existing tests. After every milestone, update agent.md in the SAME PR/commit. Never claim PASS for untested functionality. Never commit API keys or actual customer images. Keep changes scoped; do not break existing working features.

**Repository:** https://github.com/f1r10/local-image-search
**Owner:** f1r10
**Date started:** 2026-10-26
**Scope:** AWS Ubuntu (2–4 vCPU, no GPU), visual search in 100–200 products with multiple catalog photos per product. User uploads ordinary phone photo; find ranked candidates, price, brand, SKU. Priorities: speed, correctness, minimal cost, self-hostable.

## Architecture decisions
- FastAPI serving HTML/JS frontend and JSON REST endpoints; SQLite for metadata and float32 embeddings; uploaded JPEG files on persistent volume.
- Providers deliberately separated by versioned index: `lite-rgb-v1` (non-AI offline smoke test), `gemini-embedding-2-768` (main production candidate; requires a Gemini key), `facebook/dinov2-small-384` (CPU alternative, optional heavy dependencies).
- Precompute catalog embeddings at upload; query only embeds one image then performs NumPy dot product. Best image score per product; up to 5 products displayed. No universal calibrated match threshold; treat all as candidates.
- No requirement for FAISS/Qdrant at <~1000 product images. `python -m app.reindex` is required after switching providers.
- OpenAI API is **not** the primary embedding pipeline; OpenAI text embedding models cannot directly embed product photos. OpenAI Vision reranking is an optional future enhancement, not implemented.

## Milestones and current verification (do not infer success)
- M0 Repository initialization: created initial README commit e4b71caa36025e4960dcc399d402bd1f89d7160c.
- M1 MVP implementation: **PASS (for technical MVP / Lite synthetic test only)**. Routes, backend, UI, Docker Compose, provider adapters, reindex CLI and tests committed at d89bdc2560f1b545c3cfcb9c5910064bd4921849; corrective commit 4ab3285425f8f048edf8a9f36e3f1c438d2ba63b.
- M1 CI run #1 https://github.com/f1r10/local-image-search/actions/runs/40283239747 FAILED during collection (ModuleNotFoundError: app). Fixed runner invocation to python -m pytest and SQLite connection lifetime in 4ab3285.\n- M1 CI run #2 https://github.com/f1r10/local-image-search/actions/runs/40283403488 **PASS: 5/5 tests, compileall PASS, 1 upstream Starlette/httpx deprecation warning**. Evidence: health/empty index, add and reencoded-JPEG visual retrieval, admin token guard, invalid/oversize image rejection, delete cascade. This is a synthetic red-vs-blue demo, NOT a real-world same-watch/phone accuracy benchmark.
- M1 Documentation handoff: exact user-requested lowercase `agent.md` created as canonical progress log, with `AGENTS.md` as discoverable agent-tool entrypoint (commit 0203a8e1a1d454388880c73d628b02d0159d2376). Related full CI run: https://github.com/f1r10/local-image-search/actions/runs/40283857024 **PASS** (5/5 API tests, Python compile, Docker build, HTTP smoke).\n- M1 Docker container smoke: **PASS** at https://github.com/f1r10/local-image-search/actions/runs/40283647486 (commit f55bdf3662374df92a6734094873896637ee1e94): 5/5 API tests PASS, Python compileall PASS, Docker image build PASS, container startup + /api/health + web / over HTTP PASS. This validates GitHub Actions runner, not AWS server or Gemini/DINOv2 inference.\n- M2 AWS deployment: NOT STARTED; server access not supplied. Instructions available in README.
- M3 Real-world phone image evaluation: NOT STARTED. No user-owned photos provided. Collect 20–30 products with diverse views and evaluate Top-1, Top-3, P50/P95 latency.
- M4 Production hardening: NOT STARTED; rate limits, backups, admin auth improvements, SKU constraints, observability, API quota control, model fallback / timeouts.

## Security checklist
- `.env` and `data/` ignored; ADMIN_TOKEN mandatory for writes, constant-time comparison.
- Validate image decodability/type/6MB file size/20MP pixel limit; EXIF orientation normalized; JPEG re-encoded without EXIF metadata. Persistent volume owned by unprivileged container user.
- Public search is NOT rate-limited yet and could consume Gemini quota. DO NOT expose publicly without rate limits and abuse budget control. Use Nginx HTTPS proxy before testing on internet.
- No claims of similarity as calibrated probability or exact model identity.

## Exit criteria for first test
1. Tests pass on CI and can create/search/delete product with re-encoded JPEG image.
2. Invalid or oversize uploads rejected; unauthenticated writes rejected.
3. Health endpoint reports provider/index state.
4. Real Gemini embeddings and real-world product phone photos remain independently pending until API key/test data available.

## Required next actions after MVP commit
1. [DONE for initial MVP] Inspect GitHub Actions; initial failure fixed, 5/5 synthetic integration tests PASS (run 40283403488). Future commits must retain CI PASS.
2. [NEXT] Obtain AWS host access or have user run the documented deployment; never assume deployment. Current CI proof is https://github.com/f1r10/local-image-search/actions/runs/40283647486 (all jobs successful).
3. Set Gemini key in server `.env` (never in Git) and explicitly reindex existing products.
4. Collect real images for catalog+phone benchmark (different angles, backgrounds, similar watch models).
5. Determine error threshold using actual evaluation data; add safe no-match behavior and reranker when justified.
6. Add HTTP rate limiting, backup/restore test, HTTPS hardening, CPU throughput benchmark, Docker DINOv2 profile.

## Working agreements
- Keep provider names and embedding versions stable, and migrate/reindex before mixing indexes.
- Every stage update: date, commit/PR ID, files touched, PASS/FAIL/NOT RUN test evidence, known defects, next priority.
- Review agent.md at the start of every separate conversation, even when user only writes "davam et".


## M2a — 2026-10-26: Colab YOLOv8n + DINOv2 CPU integration (feature branch)
- **User-supplied working experiment:** YOLOv8n crop + facebook/dinov2-small mean pooling + L2 normalization + cosine retrieval; Colab result subjectively satisfactory. User supplied no catalog/query images or actual elapsed milliseconds to this conversation.
- **Branch:** feat/visual-retrieval-pipeline, base main f9d36307d929323d90c91308577121bce25da488.
- **Implementation:** new app/vision_pipeline.py; `dinov2-yolo-torch` and experimental `dinov2-yolo-onnx` providers in app/embedding.py; dedicated Dockerfile.cpu, compose.cpu.yaml, requirements-vision-cpu.txt and requirements-vision-onnx.txt; safe crop/mean pooling unit tests. Old Lite/Gemini/DINO CLS providers preserved. Main API concurrency configurable via EMBEDDING_CONCURRENCY.
- **Important behavior:** COCO YOLOv8n does not detect wristwatches as a native category. Ignore person classes, crop only configured known products, otherwise use entire image. This may differ from the Colab naive highest-confidence crop; compare results with user's original data.
- **Index identity:** Embedding provider key includes backend and crop configuration fingerprint. Provider/crop changes require a full reindex; never compare incompatible vectors.
- **Security and deployment:** default Docker remains lightweight; CPU Docker separate and potentially memory/disk-heavy. Persistent model-cache volume; no API key needed in CPU mode. Not deployed to user's AWS; no host credentials supplied.
- **Licensing:** Ultralytics YOLO software/model is AGPL-3.0 or Enterprise; evaluate compliance before proprietary/commercial use. DINOv2 Small model card Apache 2.0.
- **Test evidence at coding stage:** NOT RUN for heavy-model weights/ONNX export/AWS CPU benchmarks or real watch accuracy; prior Lite CI was PASS at run 40283647486. New unit regression tests will be checked by the PR GitHub Actions run; do not mark PASS before checking it.
- **Next priority:** 1) inspect PR CI; fix any failures; 2) deploy CPU image on 2–4 vCPU/4–8 GB instance; 3) test Torch end-to-end; 4) test ONNX export and speed against Torch; 5) compare crop on/off with 20–30 representative user photos; 6) tune/validate no-match threshold; 7) rate-limit public endpoints and back up SQLite + JPEGs.
