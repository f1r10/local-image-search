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

Məlumatları qorumaq üçün `data/` və `.env` Git-ə daxil edilmir. Admin açarı heç vaxt repozitoriyaya daxil edilməməlidir. `AGENTS.md` inkişaf planı və sessiyalararası həqiqət mənbəyidir: **hər yeni sessiyada ilk onu oxuyun**.

### Mənbələr
- [Gemini multimodal embedding API](https://ai.google.dev/gemini-api/docs/embeddings)
- [Meta DINOv2 Small](https://huggingface.co/facebook/dinov2-small)
