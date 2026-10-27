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

- **Real-weights CPU smoke verification:** PR workflow `.github/workflows/vision-cpu.yml` introduced; downloads CPU-only PyTorch, YOLOv8n and DINOv2 weights, checks 384-d normalized vector and repeated-inference consistency. At submission: **PENDING**; check run status before describing as PASS. This synthetic test cannot measure watch retrieval accuracy or AWS latency.

- Verified test 2026-10-26: real CPU inference PASS https://github.com/f1r10/local-image-search/actions/runs/40286503050 (initial 307.9 ms, warm 185.6 and 183.9 ms, synthetic image; 384-dimension normalized vectors). Full CI PASS https://github.com/f1r10/local-image-search/actions/runs/40286503061. AWS timing and ONNX performance NOT TESTED.

- ONNX follow-up (2026-10-26): first attempt failed due to an Optimum runtime compatibility issue (CI 40286932427). Replaced with direct ONNX Runtime; smoke PASS (CI 40287245273). Tested embedding 99.3 ms without YOLO crop on hosted CPU, synthetic image only. AWS and real-product benchmarks remain pending. Reindex on ONNX version changes.

## M2 integration outcome — 2026-10-26
- PR #1 was squash-merged into main at 457d343c64a784781a8a3ec8105bbd300f0bad65. Feature adds Colab-compatible YOLOv8n + DINOv2 Small mean-pooling as optional CPU Torch/ONNX providers, preserving existing Lite/Gemini provider paths.
- Latest PR full CI PASS: https://github.com/f1r10/local-image-search/actions/runs/40287532578. Real model CPU smoke PASS (both Torch and ONNX): https://github.com/f1r10/local-image-search/actions/runs/40287532555.
- Previous ONNX library issue was fixed by using direct onnxruntime with one-time torch.onnx.export. ONNX index uses onnx-v2 identity; provider changes require reindex.
- Quantitative runner-only synthetic smoke: Torch warmed 185/184 ms with YOLO auto; ONNX 99 ms with YOLO off. NOT comparable directly, and NOT AWS results.
- STILL OPEN: AWS installation, genuine watch database/phone-photo tests (Top-1, Top-3, no-match), measured AWS latency, public endpoint rate limits/HTTPS, licensing review before commercial deployment.
- New sessions MUST read agent.md and README.md and confirm main HEAD before any further edits.


## Module 2 exploratory research and Colab prototype — 2026-10-26
- Goal: reverse product photo search across the public web, independently from Module 1 local catalog.
- Reviewed Google Cloud Vision WEB_DETECTION, SerpApi Google Lens, SearchApi, RapidAPI, ddgs and multi-engine browser handoff. No verified stable unlimited free automated internet-wide reverse-image API. SerpApi publishes 250 free monthly searches; Google Cloud Vision first 1000 Web Detection units/month free, billing required.
- Colab prototype prepared as a downloadable artifact: Module2_Internet_Image_Search_Colab.ipynb with two vendor adapters and opt-in ddgs text-image fallback; not committed to main until real user-run Colab evaluation.
- Offline tests: 8/8 PASS for provider response parsing, upload/search request construction, image transcoding and guards; notebook mock smoke PASS. These are not live external provider results.
- Live Colab / SerpApi / Vision search: NOT RUN (keys and real test images unavailable in this session).
- Product search quality and AWS deployment: NOT RUN.
- Integration gate: user runs the notebook with real phone images and shares anonymized JSON results or Top-1/Top-3 findings; then implement Module 2 in a separate FastAPI route/UI. Keep Module 1 provider indexes and existing tests intact; add key management, timeout, rate limit, per-provider budget, caching, SSRF protections and new regression tests.
- NOTE: Notebook artifacts currently delivered in the conversation, not yet placed in repository.

## Module 2: no-key alternatives — 2026-10-26
- User rejected SerpApi's 250 free searches/month as too restrictive. API-key-free does NOT imply guaranteed unlimited external searching.
- Research: kitUIN/PicImageSearch (MIT; Google Lens, Yandex, Bing clients without SerpApi credits); omkarcloud/google-lens-scraper (MIT, self-hosted, but README acknowledges Google CAPTCHA and IP throttling after tens of searches, possible paid proxy/CAPTCHA); taazkareem/google-lens-pro (part free/part paid, may need Google session). SearXNG is text/image-keyword search fallback, NOT true upload-based reverse-image search.
- Separate Colab experiment Module2_Keyless_Reverse_Search_Colab.ipynb prepared in conversation: notebook syntax/format validated; one manual image upload; three no-key PicImageSearch engines, serial searches and timing/error JSON report. NOT committed to app and live Colab provider search NOT RUN.
- Pending: test real phone images in Colab, observe rate limits and precision, then implement separate Module 2 route/UI and regression tests while retaining Module 1.


## Module 2 Yandex parser debug — 2026-10-26
- User ran own Windows Python requests reverse-image upload with TEST2.jpeg: received cbirId successfully and a tags guess, but extracted only a /get-images-cbir/.../ocr link and 0 source sites.
- Confirmed parser issue: page data-state normally contains initialState.cbirSites.sites, not top-level cbirSites; other useful paths initialState.cbirSitesList.sites, cbirSimilar.thumbs, cbirMarketProducts.products, cbirTags.tags. The /ocr image URL is not a matched source page.
- External evidence: kitUIN/PicImageSearch/PicImageSearch/model/yandex.py and ScrapingBee/yandex-reverse-image-api README (documented 2026-09 live). Yandex may return a CAPTCHA HTTP 200 or no matching sites; never assume a result count of zero implies absence online.
- Delivered local Windows-ready standalone diagnostic script test1_fixed.py + test_parser.py and README_AZ.md as user-downloadable ZIP, **not** merged into production app. 5 deterministic offline unittest cases PASS, Python compile PASS. Real TEST2.jpeg/live Yandex result NOT RUN because original image was not supplied.
- Next: User runs the script locally and inspects yandex_debug/results.json fields (status/title/captcha/state_keys/result counts). Do not share private raw HTML/session tokens. Once validated on real images, implement module2 as separate provider route and UI, with cache/rate limits and timeouts; keep Module1 unchanged.


## Module 2 Yandex HTML DOM fallback — 2026-10-27
- User supplied a real browser DOM anchor: href points to `https://okprint.az/wp-content/uploads/2022/10/metal-qelem-kod-114.webp`, `img[src]` is an `avatars.mds.yandex.net` thumbnail; alt contains an Okprint pen title. The original image URL is NOT evidence of a store/product webpage URL. A second XPath points to another sibling anchor, but its href was not supplied.
- Reproduced user-provided HTML fragment in deterministic tests. Prepared downloadable standalone `test1_fixed_v3.py`, `test_parser_v3.py`, `README_AZ.md` in conversation ZIP (NOT integrated into production). Fix extracts `dom_images` (image_url, optional page_url, thumbnail, title) from ImagesApp list-item anchors as a fallback while retaining `initialState.cbirSites.sites` JSON parsing.
- Offline unit tests 6/6 PASS and Python compile PASS: exact user anchor, separate second-anchor page URL, initialState JSON, dedup, navigation filtering, CAPTCHA detection.
- Live Yandex TEST2.jpeg / response body NOT RUN: user provided browser HTML but no local image or raw requests response. Distinguish JS-rendered DevTools Elements from Network Document Response; `requests` cannot see JS-only DOM. Never treat absent DOM/JSON results as definite no-match or bypass CAPTCHA.
- Next: user runs v3 with TEST2.jpeg and inspects `yandex_debug/results.json` for sites, dom_images, page_url and diagnostics. Keep main Module 1 unchanged; only integrate Module 2 once live provider results validated.


## Module 2 Yandex Colab v4 visual prototype — 2026-10-27
- User-provided real run of preceding parser: successful cbirId, 4 Yandex site pages, 2 DOM image matches. Source image `metal-qelem-kod-114.webp` corresponded to `https://okprint.az/tag/promo-qelemler/`; distinct `metal-qelem-kod-115.jpg` corresponded to `https://okprint.az/qelem-metal-kod-115/`. Never conflate 114 photo with 115 product page.
- Created a self-contained Colab notebook `Module2_Yandex_Image_Search_Visual_Colab.ipynb` in conversation outputs, plus standalone `yandex_search_v4.py`, `test_yandex_v4.py`, README and visual HTML sample, bundled as Module2_Yandex_Colab_v4_Package.zip. These artifacts are NOT integrated into main app; available as conversation downloads for user Colab evaluation.
- Improvements: Yandex uploaded JPEG normalization/EXIF stripping; parse initialState.cbirSites and same-li HTML image/page pairs; prefer proven JSON image-to-page pair, never fabricate source page; deduplicate; tags, similar products; robust 403/429/CAPTCHA diagnostics without bypass; safe HTML escaping/http(s)-only URLs; polished card gallery with image/title/original image link/source page and HTML+CSV+JSON exports.
- Evidence: local unit tests 13/13 PASS (fixtures, fake HTTP session, URL sanitation, association, HTML render, upload, export); Python compile PASS, nbformat.validate PASS and Python notebook cells AST PASS. Live notebook Colab/Yandex search NOT RUN because user image/runtime unavailable in assistant environment.
- User requested first real Colab evaluation, THEN separate Module 2 GitHub app integration. Do not implement production Modul 2 APIs/UI until user reports test; do not alter Module 1.
- Caveat: keyless Yandex is unofficial, subject to rate limiting, CAPTCHA and terms; no guaranteed unlimited free production operation. Next run Colab against pen and several other product photos, inspect candidate URLs and latency, then implement separate module2 route with caching, rate limits, clear failures and security safeguards.


## Module 2: Google Lens visual Colab prototype — 2026-10-27
- User asked for equivalent of Yandex reverse-image product search against Google. Correct target is Google Lens photo upload, rather than keyword-only Google Images.
- Evaluated current kitUIN/PicImageSearch `GoogleLens` adapter (supports local `file` upload, search modes all, visual_matches, exact_matches and products); official Google Help confirms similar images and source websites can be returned.
- Built standalone Google Colab notebook `Module2_Google_Lens_Visual_Colab.ipynb` and reusable `google_lens_visual.py`, `test_google_lens_visual.py`, README and demo HTML, bundled as Module2_Google_Lens_Colab_Package.zip in conversation artifacts. NOT integrated in the repository app; user is testing notebook first.
- Key difference vs Yandex: PicImageSearch GoogleLens items contain a source-page URL and thumbnail, but no guaranteed original full-resolution image URL. Prototype deliberately leaves `original_image` blank rather than fabricating it.
- Colab implementation uses top-level `await search_google_lens_async` and async GoogleLens/Network, because PicImageSearch.sync may produce an unawaited coroutine in an existing notebook event loop. One requested mode per request; no automatic retry or CAPTCHA bypass.
- Protections: JPEG conversion, EXIF strip, source max 12 MB and Pillow decompression limits, http(s)-only result links, HTML escaping, CSV spreadsheet formula mitigation, verified page-vs-thumbnail distinctions, deduplication, JSON/CSV/HTML export.
- Evidence: 14/14 deterministic local unittest PASS incl. mocked async external call and no false original-image URL; Python compile and notebook structural validation PASS. DEMO HTML is synthetic; live Google/Colab search NOT RUN, no access to user's actual photo.
- Google Lens Shopping results are geographically restricted (Azerbaijan not on Google's published supported list); all/visual_matches recommended. No stable guaranteed unlimited free automated Google Lens service. Colab/shared IPs may be challenged.
- Integration gate: user runs Colab on real product photos and reports results, time, blocked/error behavior, then develop separate Module 2 Google adapter route/UI; preserve Module 1 and all CI. Add caching, limits, auth and abuse protection before server exposure.
