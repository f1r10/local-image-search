# Visual Product Finder

100–200 məhsulun bir və ya bir neçə kataloq şəkli ilə telefon şəkilləri arasında vizual oxşarlıq axtarışı. AWS 2–4 vCPU, GPU-suz server üçün hazırlanmış ilkin MVP.

**Vacib:** V1-də göstərilən oxşarlıq balı məhsulun dəqiq model kimi tanındığına dair ehtimal deyil. İlk 5 nəticə yalnız namizədlərdir. Telefonla çəkilmiş real məhsullar üzərində keyfiyyət sınağı hələ aparılmalıdır.

## Provider seçimi

- `IMAGE_PROVIDER=lite` — heç bir API açarı olmadan, Pillow+NumPy ilə dərhal işləyən **texniki demo**. AI deyil; fərqli rakurs, arxa fon və çox oxşar model tanınmasına güvənməyin.
- `IMAGE_PROVIDER=gemini` — tövsiyə olunan əsas rejim. `GEMINI_API_KEY` tələb edir. Rəsmi Google `gemini-embedding-2` şəkil embedding-ləri, 768 ölçülü vektor; API kvotaları və şəbəkə gecikməsi tətbiq olunur.
- `IMAGE_PROVIDER=dinov2` — tam lokal, CPU rejimi. `facebook/dinov2-small` modeli, PyTorch və Transformers tələb edir. Dockerfile standart olaraq ağır Torch paketlərini quraşdırmır; lokal Python/virtualenv ilə `pip install -r requirements-dinov2.txt` və əvvəlcədən model endirilməsi tələb olunur. Serverdə performans ayrıca ölçülməlidir.

Provider dəyişdikdə **mütləq bütün kataloqu yenidən indeksləyin**. İki providerin vektorları uyğun deyil. Cari rejim üçün `/api/health` səhifəsində `index_ready` yoxlanılır.

## İlk dəfə AWS/Ubuntu-da quraşdırma

```bash
git clone https://github.com/f1r10/local-image-search.git
cd local-image-search
cp .env.example .env
sed -i "s/^ADMIN_TOKEN=.*/ADMIN_TOKEN=$(openssl rand -hex 32)/" .env
chmod 600 .env
mkdir -p data
sudo chown -R 10001:10001 data
docker compose up -d --build
curl http://127.0.0.1:8000/api/health
```

Server üzərində ilk test üçün `http://127.0.0.1:8000/` açılır (SSH tuneli: `ssh -L 8000:127.0.0.1:8000 ubuntu@SERVER_IP`). Xaricdən giriş üçün Nginx və HTTPS reverse proxy qurun. 8000 portu compose-da qəsdən yalnız localhost-a bağlanıb. `ADMIN_TOKEN` dəyərini serverdə `.env` faylından götürüb veb panelə daxil edin. Şifrəni GitHub-a və söhbətə göndərməyin.

### Gemini rejimi

Google AI Studio-dan API açarı alın, `.env`-də `IMAGE_PROVIDER=gemini` və `GEMINI_API_KEY=...` yazın. `docker compose up -d` ilə yenidən başlayın. Əvvəl Lite rejimində məhsullar əlavə edilibsə, Gemini rejimində reindex lazımdır:

```bash
docker compose exec image-search python -m app.reindex
```

Bu əməliyyat API sorğuları yaradır, pul/kvota xərcləyə bilər və bir qədər vaxt ala bilər. Reindex fasilədə edilərkən bəzi şəkillər əvvəlki provider indeksində qalacaq; `/api/health` cavabında yoxlayın. Eyni anda bir neçə reindex başladılmamalıdır.

### Docker-suz lokal test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export ADMIN_TOKEN=test-local
export IMAGE_PROVIDER=lite
uvicorn app.main:app --reload
# ayrı terminal: python -m pytest -q
```

## Tətbiq funksiyaları

- Məhsul əlavə etmək: brend, ad, SKU, qiymət, qısa təsvir və 1–6 şəkil.
- Brauzerdə telefon kamerasından çəkilən və ya cihazdan seçilən şəkillə axtarış.
- Ən oxşar məhsulların bir şəkil üzrə ən yaxşı vektor nəticəsinə görə sıralanması, qiymət və SKU göstərilməsi.
- Admin açarı ilə məhsul yaratma/silmə; API açarı serverdə saxlanılır, brauzerdə daimi saxlanılmır.
- SQLite məlumat bazası və lokal JPEG faylları, Docker volume ilə qalıcı saxlanma.
- `/api/health` ilə indeks/provayder vəziyyəti, `python -m app.reindex` ilə yenidən indeksləmə.

## Məhdudiyyətlər və təhlükəsizlik

Real məhsul şəkillərinin test bazası, Top-1/Top-3 dəqiqlik ölçümü və AWS-də canlı deploy bu commit-in bir hissəsi deyil. Gemini və DINOv2 üçün canlı inferensiya da ayrıca təsdiqlənməlidir. Public internetə açmazdan əvvəl HTTPS, Nginx rate limit, auth/abuse qoruması və backup əlavə edilməlidir. Gemini rejimində göndərilən şəkillər xarici Google API-yə ötürülür.

Məlumatları qorumaq üçün `data/` və `.env` Git-ə daxil edilmir. Admin açarı heç vaxt repozitoriyaya daxil edilməməlidir. `agent.md` inkişaf planı və sessiyalararası həqiqət mənbəyidir: **hər yeni sessiyada ilk onu oxuyun**. Standart agent inteqrasiyaları üçün `AGENTS.md` yönləndirici fayl da saxlanılır..

### Mənbələr
- [Gemini multimodal embedding API](https://ai.google.dev/gemini-api/docs/embeddings)
- [Meta DINOv2 Small](https://huggingface.co/facebook/dinov2-small)


## Colab YOLOv8n + DINOv2 Small pipeline (optional CPU mode)

**Status:** integrated as a selectable experimental provider; basic CI checks do NOT download or execute neural-model weights. The Colab user reports promising quality, but AWS timing and real watch accuracy are not yet independently measured.

New providers: `dinov2-yolo-torch` (PyTorch CPU, safest initial port of the Colab test) and `dinov2-yolo-onnx` (experimental ONNX Runtime). They use mean pooling of `last_hidden_state` across all tokens, followed by L2 normalization. The older `dinov2` provider still uses CLS pooling and is unchanged. **Never mix embeddings from two configurations.** Model/provider/crop config changes require `python -m app.reindex`.

**YOLO caveat:** pre-trained `yolov8n.pt` uses COCO80; there is no wristwatch class. The highest-confidence box may be a *person* instead of a watch. This implementation filters product classes and falls back to the original image unless an eligible object is found. For an all-watch catalog compare `YOLO_CROP_MODE=off` (lower CPU latency) versus `auto` using real photos. Cropping is applied in the same way to catalog and query images.

### CPU Docker run on AWS (2–4 vCPU; 4–8 GB RAM strongly recommended)

```bash
git pull --ff-only
cp .env.example .env  # only if first setup; NEVER overwrite a working .env
# Ensure ADMIN_TOKEN already exists (see earlier instructions). Do not paste secrets in chat.
mkdir -p model-cache data
sudo chown -R 10001:10001 model-cache data
docker compose -f compose.yaml -f compose.cpu.yaml build
docker compose -f compose.yaml -f compose.cpu.yaml up -d
docker compose -f compose.yaml -f compose.cpu.yaml logs --tail=100 image-search
curl http://127.0.0.1:8000/api/health
docker compose -f compose.yaml -f compose.cpu.yaml exec image-search python -m app.reindex
```

The optional CPU image is much larger than the default image (PyTorch, Ultralytics, Transformers and ONNX dependencies). The first inference downloads YOLO/DINO weights or exports ONNX into the persistent `model-cache` volume; provide sufficient disk space and startup time. Use `IMAGE_PROVIDER_CPU=dinov2-yolo-onnx` in `.env` to try ONNX; ONNX export has not been validated on the target AWS host. Start with Torch to reproduce Colab behavior. Docker Compose uses only localhost port 8000 as before.

For watch-only benchmarks, set `YOLO_CROP_MODE=off` and **reindex** before comparisons. Record separate Top-1/Top-3, P50/P95 latency, cold-start, memory, and query image conditions. An embedding cosine score is not a calibrated confidence percentage.

**Licenses:** DINOv2 Small is Apache-2.0, while Ultralytics YOLOv8/code/weights have AGPL-3.0 or paid Enterprise terms. Commercial or closed-source SaaS deployment needs an explicit license compliance review; do not assume a public GitHub repository alone satisfies all obligations. See https://www.ultralytics.com/license and https://huggingface.co/facebook/dinov2-small.

**Tests:** `python -m pytest -q` checks deterministic cropping, mean pooling and existing API behavior without heavyweight model downloads. A full CPU-model inference smoke test and real-phone retrieval benchmark remain **NOT RUN** until an appropriate host/model download and sample images are available.
