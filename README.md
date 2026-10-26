# EmbedEye — Local-First Semantic Photo & OCR Search

**EmbedEye** is a high-performance, 100% privacy-first local semantic photo search and document OCR intelligence application optimized for budget GPUs (such as the **NVIDIA RTX 3050 4GB VRAM Laptop GPU**, 16GB RAM) with seamless automatic fallback to **CPU** when CUDA is unavailable.

> 📖 **Architecture & Code Structure Guide**: For a detailed technical walkthrough of the Object-Oriented design patterns (Strategy, Factory, Singleton, DAO, Context Manager), module relationships, class hierarchies, and execution sequence diagrams, see [CODE_STRUCTURE.md](file:///e:/projects/AI%20image%20searcher/CODE_STRUCTURE.md).

Powered by **OpenCLIP** (512-d visual embeddings), **Microsoft Florence-2** (unified detailed scene captioning + OCR in a single model), **RapidOCR** (CPU-only document verification), **mixedbread-ai Cross-Encoder** (local neural reranking), and **Qdrant** (embedded local vector database).

All model weights are stored strictly inside the project's `./models/` directory for full offline portability without polluting user or global directories.

---

## 🌟 Key Features

- **100% Local & Offline**:
  - Zero external APIs, zero online LLM calls, zero network dependencies.
  - No API keys needed; private photos and search queries never leave your machine.
- **Strict Single-Model GPU Isolation (`src/gpu_manager.py`)**:
  - Only **ONE** heavy model resides on the GPU at any time during indexing.
  - Context-managed lifecycle (`with manager.use(...)`): models are loaded on demand and immediately unloaded with thorough garbage collection, CUDA synchronization, and cache clearing.
  - At search time, OpenCLIP and the Cross-Encoder Reranker are permitted to coexist (~1.5GB total VRAM), while Florence-2 is strictly prohibited at search time.
  - Automatic OOM recovery: catches CUDA Out-of-Memory errors, halves batch size, retries, and falls back to CPU if single items fail.
- **Unified Captioning & OCR (Microsoft Florence-2)**:
  - Replaces legacy BLIP captioners with **Florence-2-large** (or **Florence-2-base**).
  - Simultaneously produces detailed descriptive captions (`<MORE_DETAILED_CAPTION>`) and extracts text found inside photos (`<OCR>`) in one model pass.
  - Cleans raw OCR output (filters repetitive artifacts and garbage characters, discards noise with < 4 alphanumeric characters, caps at 1000 chars).
- **CPU-Only Document & Screenshot Verification (RapidOCR)**:
  - Automatically runs RapidOCR on CPU (ONNX Runtime, 0 VRAM) on candidate screenshots and documents.
  - Updates OCR text if more comprehensive and confident than Florence-2.
- **Visual & Keyword Hybrid Search**:
  - Semantic natural language queries powered by OpenCLIP (`ViT-B-32`, `laion2b_s34b_b79k`).
  - Quoted queries (e.g. `bill from 'Reliance'`) execute full-text matching against the Qdrant `ocr_text` payload index and merge keyword hits into the candidate pool.
  - Cross-encoder reranker evaluates `"Caption: {caption} | Text in image: {ocr_text[:200]} | Date: {date}"`.
  - Guarantees CLIP rank #1 is always preserved in the final top 4 results.
- **Fault-Tolerant Staged Indexing with SQLite Cache**:
  - Persists intermediate progress per stage to `./data/indexer_cache.db`.
  - If indexing is interrupted, restarts continue seamlessly without redoing finished stages.
  - Includes `--recaption` mode to upgrade existing libraries with Florence-2 without re-embedding CLIP vectors.

---

## 🔄 Staged GPU Pipeline

```
+-----------------------------------------------------------------------------------+
|                            IMAGE DIRECTORY SCAN                                   |
|                (Path & mtime check: processes only new / modified files)          |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|  STAGE A: OpenCLIP Image Embedding                                                |
|  - Load OpenCLIP ViT-B-32 (fp16 on CUDA, ~0.4 GB)                                  |
|  - Compute 512-d normalized image vectors in batches of 16-32                     |
|  - Cache vectors in RAM & SQLite                                                  |
|  - UNLOAD OpenCLIP -> gc.collect() -> torch.cuda.empty_cache() (VRAM returns to 0)|
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|  STAGE B: Florence-2 Captioning + OCR                                             |
|  - Load Florence-2-large (fp16 on CUDA, ~1.5 GB)                                  |
|  - Run <MORE_DETAILED_CAPTION> + <OCR> in batches of 2-4                          |
|  - Post-process and clean OCR text                                                |
|  - Cache captions & OCR text in RAM & SQLite                                      |
|  - UNLOAD Florence-2 -> gc.collect() -> torch.cuda.empty_cache() (VRAM returns to 0)|
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|  STAGE C: RapidOCR Verification (Optional, USE_RAPIDOCR=true)                     |
|  - Runs strictly on CPU via ONNX Runtime (0 VRAM consumed)                        |
|  - Evaluates images with detected text or document/screenshot aspect ratio        |
|  - Merges confident text into ocr_text payload                                    |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|  STAGE D: Local Qdrant Upsert                                                     |
|  - Batch upsert points in chunks of 64                                            |
|  - Stores 'clip' 512-d vector + payload: caption, ocr_text, has_text, date, mtime  |
|  - Checkpoints completed status in SQLite cache                                   |
+-----------------------------------------------------------------------------------+
```

---

## 📊 VRAM Budget & Hardware Profile (RTX 3050 4GB)

The RTX 3050 Laptop GPU provides **4096 MB VRAM**. The architecture ensures that models are never loaded simultaneously during indexing:

| Operation | Resident Model(s) | Precision | Allocated VRAM | Peak VRAM | Headroom Remaining |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Indexing: Stage A** | OpenCLIP ViT-B-32 | fp16 | ~350 MB | ~630 MB | **~3460 MB** |
| **Indexing: Stage B** | Florence-2-large | fp16 | ~1480 MB | ~1500 MB | **~2590 MB** |
| *Indexing: Stage B (alt)* | *Florence-2-base* | *fp16* | *~450 MB* | *~700 MB* | *~3390 MB* |
| **Indexing: Stage C** | RapidOCR (CPU) | int8/fp32 | **0 MB** (Host RAM) | 0 MB | **4096 MB** |
| **Indexing: Stage D** | Qdrant Client | N/A | **0 MB** (Host RAM) | 0 MB | **4096 MB** |
| **Search Time** | OpenCLIP + Reranker | fp16 | ~1185 MB | ~1205 MB | **~2890 MB** |

> **Safety Guarantee**: Peak VRAM during indexing never exceeds **1.5 GB**, leaving more than 2.5 GB of safety headroom on a 4GB VRAM card.

---

## 🏗️ Architecture & Project Structure

```
embedeye/
  models/                 # ALL model weights stored locally here (git-ignored)
    open_clip/            # OpenCLIP ViT-B-32 weights
    transformers/         # Microsoft Florence-2 weights
    reranker/             # mixedbread-ai/mxbai-rerank-large-v1 cross-encoder weights
    sentence_transformers/# all-MiniLM-L6-v2 CPU weights
    tesseract/            # Standalone Tesseract v5.5.3 binaries & tessdata
    torch/                # Torch core cache
    huggingface/          # HuggingFace hub cache
  data/
    qdrant/               # Embedded Qdrant local vector database
    indexer_cache.db      # SQLite checkpoint database for visual indexing
    ocr_indexer_cache.db  # SQLite checkpoint database for OCR indexing
  scripts/
    download_models.py    # Downloads all models into ./models/ with size breakdown
    verify_ocr_mode.py    # Automated test suite for OCR indexing and search
  backend/
    server.py             # Uvicorn launcher, ready line handshake, ephemeral port/token
    api.py                # FastAPI routes (health, status, folder summary, jobs, websockets)
    jobs.py               # Single-job worker runner with cancellation & GPU isolation
    events.py             # Thread-safe event bus & WebSocket broadcaster
    security.py           # Bearer token auth, origin protection, path sanitization
    staging.py            # SQLite staging store (./data/staging.db)
  gui/
    EmbedEye.Gui/         # Avalonia 11 desktop app (.NET 8, CommunityToolkit.Mvvm)
      ViewModels/         # MainWindowViewModel, ViewModelBase
      Views/              # MainWindow, SplashView (Mica/Acrylic, Fluent design)
      Services/           # JobObjectService, BackendProcessManager, ApiClient, WebSocketClient
  src/
    cli.py                # Unified CLI interface (index, search, ocr, clear, status)
    config.py             # Paths, env overrides, hardware detection, settings
    gpu_manager.py        # Strict GPU memory manager & context manager API
    models.py             # Loaders, inference helpers, OCR cleaning, RapidOCR
    indexer.py            # 4-stage staged indexer with crash recovery
    search.py             # OpenCLIP + Reranker search + OCR keyword matching
    ocr_engine.py         # Multi-engine OCR strategies (Tesseract, RapidOCR, Florence)
    ocr_indexer.py        # Document chunking, multiprocessing, OCR indexing
    ocr_search.py         # Hybrid vector/keyword retrieval, RRF, reranking
    status.py             # Hardware telemetry, live VRAM, and collection counts
    os_utils.py           # Native desktop file reveal & explorer launcher
    thumbnails.py         # Image and PDF page preview thumbnail generator
  tests/
    test_gpu_manager.py   # Unit test asserting 50MB VRAM return to baseline
    test_backend.py       # Comprehensive pytest suite for API, security, staging, jobs
    test_e2e_pipeline.py  # End-to-end integration test of 20-image pipeline
  publish.ps1             # PowerShell script to produce self-contained win-x64 build
  .env.example            # Environment configuration template
  requirements.txt        # Verified Python package dependencies
  CODE_STRUCTURE.md       # Architecture & OOP documentation
  README.md               # Documentation and technical guide
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10 to 3.13
- Windows, Linux, or macOS
- NVIDIA GPU with CUDA drivers (e.g. RTX 3050 Laptop GPU) or modern CPU

### 2. Create Virtual Environment
```powershell
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux/macOS:
source venv/bin/activate
```

### 3. Install PyTorch with CUDA 12.4
For GPU acceleration:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

*For CPU-only machines:*
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Default configuration in `.env`:
```env
# Local Neural Reranker (100% Offline, runs locally on GPU/CPU)
RERANKER_MODEL_NAME=mixedbread-ai/mxbai-rerank-large-v1
RERANKER_BATCH_SIZE=16

# Florence-2 Model Configuration (Captioning + OCR in one model)
FLORENCE_MODEL_NAME=microsoft/Florence-2-large
FLORENCE_BATCH_SIZE=2
FLORENCE_MAX_NEW_TOKENS=256
FLORENCE_NUM_BEAMS=3

# CLIP Batch Size (fp16 on CUDA)
CLIP_BATCH_SIZE=16

# OCR Post-Processing & RapidOCR Fallback (CPU only, 0 VRAM)
USE_RAPIDOCR=true
MAX_OCR_CHARS=1000

# Offline mode: Set to 1 once downloaded to prevent web requests
HF_HUB_OFFLINE=0

# Device override (auto, cuda, cpu)
DEVICE_OVERRIDE=auto
```

---

## 📥 Downloading Models to `./models/`

Run the pre-downloader script once to download weights directly into `./models/`:
```powershell
# Download all models (OpenCLIP, Florence-2-large, Neural Reranker):
.\venv\Scripts\python.exe scripts\download_models.py

# Download Florence-2-base instead (lighter, faster):
.\venv\Scripts\python.exe scripts\download_models.py --florence-model microsoft/Florence-2-base

# Pre-download both large and base:
.\venv\Scripts\python.exe scripts\download_models.py --also-florence-base
```

---

## 💻 Interfaces & Usage

EmbedEye provides two primary interfaces:
1. **Unified Command-Line Interface (CLI)**: Terminal-first, privacy-respecting search and index management.
2. **Desktop Application**: Modern Windows native GUI built with Avalonia (.NET 8) backed by EmbedEye's local Python API backend.

> ⚠️ **Database Concurrency Notice**: Embedded local Qdrant allows only **one process** to access `./data/qdrant` at a time. Do not execute CLI index/search commands while the desktop application is running.

---

### 1. Unified Command-Line Interface (`src.cli`)

EmbedEye includes a single command-line interface `python -m src.cli`:

#### A. Visual Photo Indexing & Search
```powershell
# Index a directory of photos (OpenCLIP 512-d embeddings + Florence-2 captioning):
.\venv\Scripts\python.exe -m src.cli index "C:\Users\Admin\Pictures"

# Recaption existing index without recomputing CLIP visual vectors:
.\venv\Scripts\python.exe -m src.cli index "C:\Users\Admin\Pictures" --recaption

# Search photos with semantic queries:
.\venv\Scripts\python.exe -m src.cli search "red car on sunny beach"

# Search photos with quoted OCR text:
.\venv\Scripts\python.exe -m src.cli search "bill from 'Reliance'"

# Search with offline date parsing (e.g. 2024):
.\venv\Scripts\python.exe -m src.cli search "family vacation 2024" --smart-date
```

#### B. OCR Document Indexing & Hybrid Search
```powershell
# Index a directory of documents, receipts, screenshots, or PDFs:
.\venv\Scripts\python.exe -m src.cli ocr-index "data/sample_ocr_test" --engine tesseract

# Search OCR documents (hybrid vector + keyword full-text + cross-encoder rerank):
.\venv\Scripts\python.exe -m src.cli ocr-search "electricity bill"

# Search with exact quoted phrases:
.\venv\Scripts\python.exe -m src.cli ocr-search '"Oakridge Properties"'
```

#### C. System Telemetry & Database Management
```powershell
# Display real-time GPU VRAM, active models, and collection counts:
.\venv\Scripts\python.exe -m src.cli status

# Clear databases:
.\venv\Scripts\python.exe -m src.cli clear --photos   # Clear visual photos collection
.\venv\Scripts\python.exe -m src.cli clear --ocr      # Clear OCR documents collection
.\venv\Scripts\python.exe -m src.cli clear --all      # Clear all databases
```

---

### 2. Standalone Subsystem Modules

You can also run individual subsystem modules directly:

```powershell
# Visual Photo Indexer & Searcher:
.\venv\Scripts\python.exe -m src.indexer "C:\Users\Admin\Pictures"
.\venv\Scripts\python.exe -m src.search "red car near beach"

# OCR-Only Indexer & Searcher:
.\venv\Scripts\python.exe -m src.ocr_indexer "data/sample_ocr_test"
.\venv\Scripts\python.exe -m src.ocr_search "invoice total"

# VRAM Management Unit Test:
.\venv\Scripts\python.exe -m tests.test_gpu_manager
```

---

### 3. Desktop Application (Avalonia GUI + Python API Backend)

EmbedEye features a native Windows desktop GUI built with **Avalonia 11 (.NET 8)** using the Model-View-ViewModel (MVVM) architecture with `CommunityToolkit.Mvvm`. The GUI is backed by a local FastAPI backend running on a dedicated worker thread with strict GPU isolation and real-time WebSocket telemetry.

#### A. Architecture & Lifecycle
- **Process Ownership**: The Avalonia app launches the Python backend hidden using the venv's `pythonw.exe`. The backend binds exclusively to `127.0.0.1` on an ephemeral port with an ephemeral 32-byte cryptographic Bearer token passed via environment variables.
- **Job Object Safety**: The backend child process is assigned to a Windows Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. When the GUI closes or terminates unexpectedly, Windows terminates the backend process instantly, ensuring zero orphan processes hold GPU VRAM or the local Qdrant lock.
- **Single-Instance Mutex & Named Pipe**: A second launch with a folder argument forwards the folder path to the existing window (bringing it to the foreground) and exits immediately.
- **Exclusive DB Access**: The desktop app backend is the sole owner of `./data/qdrant/`. Do not execute CLI commands while the desktop app is running.

#### B. The 4 Pipeline Actions
1. **📄 1. OCR (Text Only)**: Extracts text from images and PDF pages using the configured OCR engine (RapidOCR / Tesseract on CPU, 0 MB VRAM) and saves results to `./data/staging.db`. No Qdrant writes occur during this stage. Unchanged files (by path + mtime) are skipped.
2. **✨ 2. OCR + Caption**: Generates detailed descriptive captions and OCR text using **Florence-2** on GPU with strict memory manager isolation, followed by RapidOCR verification for documents and screenshots. Results are saved to `./data/staging.db`.
3. **💾 3. Store in DB**: Commits staged data into local Qdrant databases:
   - OCR-only rows are chunked (~250 words) and embedded into 384-d dense vectors with MiniLM on CPU, then upserted to `docs_ocr`.
   - Caption rows are encoded into 512-d OpenCLIP vectors on GPU (model loaded, used, and immediately unloaded) and upserted to `photos`.
4. **🗑️ 4. Delete from DB**: Removes all points matching the folder path (recursively) from both collections (`photos` and `docs_ocr`), plus staging and SQLite cache entries. **Your original image and document files are NEVER deleted.**

#### C. Prerequisites & Development Run
- **Prerequisites**: .NET 8.0 SDK (installed via `dotnet-install.ps1` or Microsoft installer).
```powershell
# Run the desktop app during development:
$env:DOTNET_ROOT = "C:\Users\Admin\.dotnet"
$env:PATH = "C:\Users\Admin\.dotnet;" + $env:PATH
dotnet run --project gui/EmbedEye.Gui
```

#### D. Publishing Self-Contained Release Build
Run the automated build script to produce a standalone Windows distribution in `dist/EmbedEye/`:
```powershell
powershell -ExecutionPolicy Bypass -File .\publish.ps1
```
The resulting executable is `dist\EmbedEye\EmbedEye.Gui.exe` (ready to run with zero .NET runtime dependencies on user machines).

#### E. Windows Explorer Right-Click Context Menu Integration
You can register EmbedEye directly into the Windows right-click context menu (HKCU registry, requiring no administrator elevation):
```powershell
# Register context menu:
.\dist\EmbedEye\EmbedEye.Gui.exe --register-context-menu

# Unregister context menu:
.\dist\EmbedEye\EmbedEye.Gui.exe --unregister-context-menu
```
*Note for Windows 11*: The menu item ("Open with EmbedEye") appears under **"Show more options"** (or by pressing `Shift + F10`). You can also toggle this from the Settings Gear flyout inside the application.

#### F. Accessibility & Low-Resource Flags
- **Reduce Animations**: Toggleable via the settings gear flyout, or automatically activated when Windows system animation effects are turned off (`SPI_GETCLIENTAREAANIMATION`).
- **Software Rendering**: Pass `--software-render` if running inside virtual machines or systems with unsupported display drivers:
  ```powershell
  .\dist\EmbedEye\EmbedEye.Gui.exe --software-render
  ```

---

## 🔄 Migrating Existing Libraries (`--recaption`)

If you previously indexed photos with the BLIP captioner, you can upgrade your library to Florence-2 (adding rich captions and OCR text) without recomputing visual CLIP vectors:

```powershell
.\venv\Scripts\python.exe -m src.indexer --recaption "C:\Users\Admin\Pictures"
```

This bypasses Stage A, re-runs Stage B (Florence-2) and Stage C (RapidOCR), and updates Qdrant payloads with `caption`, `ocr_text`, and `has_text`.

---

## ⚡ Switching to Florence-2-base

To optimize for maximum indexing speed or even lower memory usage:
1. In `.env`:
   ```env
   FLORENCE_MODEL_NAME=microsoft/Florence-2-base
   FLORENCE_BATCH_SIZE=4
   ```
2. Download base weights:
   ```powershell
   .\venv\Scripts\python.exe scripts\download_models.py --florence-model microsoft/Florence-2-base
   ```
Florence-2-base requires only **~450 MB VRAM** and processes images ~2x faster while maintaining strong captioning and OCR capability.

---

## 🔧 Troubleshooting

### `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`
Florence-2 uses remote model code (`trust_remote_code=True`). In `transformers` v5.0+, `forced_bos_token_id` is deprecated and causes this error. 
- **Solution**: The project pins `transformers==4.46.3` in `requirements.txt` and pins the model revision commit hash (`21a599d4...`) in `src/config.py`. Do not upgrade `transformers` to v5.x.

### CUDA Out of Memory (OOM)
The pipeline automatically handles OOM by halving the active batch size and retrying. If your system runs other GPU-intensive tasks:
1. Reduce `FLORENCE_BATCH_SIZE=1` and `CLIP_BATCH_SIZE=8` in `.env`.
2. Switch to `microsoft/Florence-2-base`.
3. To force CPU inference, set `DEVICE_OVERRIDE=cpu` in `.env`.

### Qdrant "already accessed by another instance"
Local embedded Qdrant uses an exclusive file lock on `./data/qdrant/`.
- Only one process may use `./data/qdrant` at a time (the desktop app's backend or one CLI command).
- Do not run a CLI command while the desktop app is open. Always close the desktop application or wait for background indexing tasks to complete before running CLI commands.

---

## 📄 OCR-Only Mode (Document & Bill Search)

An optimized document and text search mode that stores **ONLY** the extracted OCR text (zero CLIP visual vectors, zero captions), drastically reducing indexing time and VRAM usage.

### When to Use It
- **Document Archives**: Invoices, electricity/water bills, medical prescriptions, tax forms, rental leases, and bank statements.
- **Digital Text**: Full-page screenshots, code editor snippets, terminal logs, and system dashboards.
- **PDF Documents**: Multi-page scanned or rendered PDFs (each page extracted individually via `pypdfium2`).
- **Low VRAM Environments**: Runs entirely on **CPU** with zero GPU VRAM consumption during indexing.

### Pipeline Architecture

```
+-----------------------------------------------------------------------------------+
|                            DOCUMENT DIRECTORY SCAN                                |
|          Supports: .jpg, .jpeg, .png, .webp, .bmp, .tiff, and .pdf pages          |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|  1. PREPROCESSING & CLEANING                                                      |
|  - EXIF rotation transpose + convert to RGB                                       |
|  - Scale longest side to max 2000px (high resolution for fine text)               |
|  - PDF page rendering via pypdfium2 (~144 dpi)                                    |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|  2. OCR ENGINE EXECUTION                                                          |
|  - Parallel multiprocessing pool (OCR_WORKERS=2-3)                                 |
|  - Engine Options (OCR_ENGINE):                                                   |
|    * tesseract: System Tesseract via pytesseract with word/line confidence filter  |
|    * rapidocr (default fallback): CPU ONNX Runtime, 0 VRAM                        |
|    * florence: Florence-2 <OCR> via ModelManager (isolated on GPU, unloaded after)|
|  - Confidence filter: drops lines with confidence < OCR_MIN_CONF (default 0.5)    |
|  - Noise rejection: DISCARDS images with total alphanumeric characters < 4        |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|  3. TEXT CHUNKING & DENSE EMBEDDING                                               |
|  - Chunking: ~250 words per chunk with 30-word overlap (one Qdrant point/chunk)   |
|  - Dense Vector: 384-d normalized vector via sentence-transformers/all-MiniLM-L6-v2|
|    (Runs strictly on CPU, 0 MB VRAM)                                              |
|  - Payload: path, filename, page, chunk_index, text, full_text_len, date, mtime   |
|  - Point ID: Deterministic UUID5 of (path + page + chunk_index)                   |
|  - Upsert to Qdrant collection 'docs_ocr' in batches of 64                        |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|  4. HYBRID SEARCH & RERANKING                                                     |
|  - Retrieval:                                                                     |
|    (a) Dense Vector Search: MiniLM 384-d cosine similarity (top 20)               |
|    (b) Keyword Full-Text Search: Exact phrase matching on "quoted terms" (top 20) |
|  - Fusion: Reciprocal Rank Fusion (RRF, k=60) merges into top 20 candidates       |
|  - Reranker: mixedbread-ai/mxbai-rerank-large-v1 (fp16 on GPU, batch size 8)     |
|  - Grouping: Aggregates chunks per image/page, returns Top 4 with <mark> snippet  |
+-----------------------------------------------------------------------------------+
```

### Accuracy Limits & Caveats
- **Handwriting**: Printed text achieves near 100% accuracy; cursive or messy handwriting has significantly lower recognition rates.
- **Non-Latin Scripts**: The default model weights are optimized for English / Latin scripts. To enable additional languages in RapidOCR or Tesseract, install the respective language trained data packs (`tessdata` for Tesseract or language models for RapidOCR) and update `OCR_LANGS` in `.env`.
- **Low Resolution**: Images where text height is under 12 pixels or heavily blurred will produce low confidence and may be discarded by the alphanumeric threshold.

### CLI Usage for OCR Mode

#### 1. Index a Folder of Documents / Images:
```powershell
.\venv\Scripts\python.exe -m src.ocr_indexer "data/sample_ocr_test"
```
Options:
- `--engine tesseract|rapidocr|florence`: Select OCR engine (default from `.env`).
- `--workers 3`: Number of parallel CPU worker processes.
- `--reindex`: Force re-indexing of unchanged files.
- `--clear`: Clear collection `docs_ocr` before indexing.

#### 2. Query Documents via Search CLI:
```powershell
# Loose natural language query
.\venv\Scripts\python.exe -m src.ocr_search "electricity bill"

# Exact quoted phrase query
.\venv\Scripts\python.exe -m src.ocr_search '"Oakridge Properties"'

# Query with candidate score inspection
.\venv\Scripts\python.exe -m src.ocr_search "invoice total" --show-candidates
```

---

## 🖥️ Windows Desktop GUI (EmbedEye.Gui)

EmbedEye includes a professional, native Windows desktop application built with **.NET 8 and Avalonia 11** using the Fluent theme. The UI is designed to enterprise standards with vector SVG iconography, high contrast readability, and zero emojis.

### Core Features
- **Instant Photo & Document Search**:
  - The default primary screen is the **Search Library** view.
  - Enter any natural language description or keyword to retrieve the **Top 3 best matching items**.
  - Displays rich result cards with thumbnail image preview, match confidence percentage badge, rank badge, date, caption, OCR text snippet, and quick actions (`Open Image`, `Show in Folder`).
  - Supports both **Visual Photo Search** (OpenCLIP + Neural Cross-Encoder Reranker) and **Document OCR Search** (Hybrid Vector/Keyword + Reranker).
- **Indexing & Library Management**:
  - Dedicated secondary tab for managing folder indexing with real-time file counters, 4 pipeline actions (OCR Text Only, OCR + Caption, Store in Database, Delete from Database), VRAM telemetry gauge, and live execution logs.
- **Safety & Process Isolation**:
  - Bound to Windows Job Object (`KILL_ON_JOB_CLOSE`), ensuring backend processes automatically terminate when the desktop GUI exits.
  - Single-instance mutex with named pipe argument forwarding (e.g. from Windows Explorer right-click context menu).

### Launching the Desktop Application
```powershell
# Run the self-contained distribution directly:
.\dist\EmbedEye\EmbedEye.Gui.exe
```

---

## 📜 Third-Party Models and Licenses

All deep learning models and OCR packages utilized in this project are open-source and verified against their official model cards and repositories:

| Model / Library | Primary Purpose | Repository / Card | License | Commercial Use |
| :--- | :--- | :--- | :--- | :--- |
| **Microsoft Florence-2** | Image Captioning & Text OCR | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | **MIT** | Yes |
| **OpenCLIP** (`ViT-B-32`) | Visual Vector Embeddings (512-d) | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | **MIT** | Yes |
| **mixedbread-ai Reranker** | Neural Cross-Encoder Reranking | [mixedbread-ai/mxbai-rerank-large-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) | **Apache 2.0** | Yes |
| **all-MiniLM-L6-v2** | Document Dense Embedding (384-d, CPU) | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | **Apache 2.0** | Yes |
| **RapidOCR** | CPU Document/Screenshot Text Verification | [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) | **Apache 2.0** | Yes |
| **Tesseract OCR** | Open-source OCR Engine | [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) | **Apache 2.0** | Yes |
| **pypdfium2** | PDF Page Rendering | [pypdfium2-team/pypdfium2](https://github.com/pypdfium2-team/pypdfium2) | **Apache 2.0 / BSD-3** | Yes |
| **Qdrant** | Local Vector Search Engine | [qdrant/qdrant-client](https://github.com/qdrant/qdrant-client) | **Apache 2.0** | Yes |
