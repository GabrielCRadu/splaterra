# Context proiect

Splat Prep: aplicatie personala care streamline-uieste preprocesarea footage-ului (video de la
camera/telefon/drona) pentru reconstructie 3D Gaussian Splatting. Automatizeaza extragerea de
cadre sharp, citeste metadata din camera + hardware-ul local si recomanda setari pentru
RealityScan si LichtFeld, orchestreaza RealityScan (alignment) headless, si expune totul printr-un
UI mobil-first accesibil de pe telefon (prin Tailscale) cat esti plecat de acasa. Un singur
utilizator (owner-ul), ruleaza pe PC-ul propriu. Fiecare etapa a pipeline-ului se porneste manual
din UI, cu progres vizibil prin polling.

# Comenzi

Verificate pe acest mediu (Windows, Python 3.14, venv in `.venv/`):

- Instalare dependinte: `python -m pip install -r requirements.txt`
- Rulare server (dev/prod local): `python run.py` — asculta pe `0.0.0.0:8000`, UI la `http://127.0.0.1:8000/`
- Teste: `python -m pytest -q` (sau `-v` pentru detalii)
- Un singur fisier de test: `python -m pytest tests/test_realityscan.py -v`

Nu exista comanda de build (frontend e static, fara build step) si nici comanda de deploy — nu
presupune una pana nu e configurata explicit. Nu exista un linter configurat inca.

Binare externe folosite prin subprocess (nu prin binding Python), asteptate pe PATH sau in locatia
default de instalare:
- `ffmpeg` / `ffprobe` — extragere cadre, citire metadata video
- `sharp-frames` — selectie cadre sharp (`pip install sharp-frames`)
- RealityScan — alignment; NU e pe PATH, e descoperit prin fallback la
  `C:\Program Files\Epic Games\RealityScan_*\RealityScan.exe`, sau la directorul indicat de
  env var `SPLATPREP_REALITYSCAN_DIR` daca instalarea e in alta parte (vezi `app/realityscan.py`).

# Conventii de cod

- Cod, commit-uri, nume de fisiere: engleza, fara diacritice.
- Continut vizibil utilizatorului (UI, mesaje in interfata): romana cu diacritice corecte.
- Fara comentarii inutile: doar cand explica un WHY neevident (ex. constrangerea FK care
  justifica sa nu scrii un stage row pentru un job inexistent — vezi `alignment_service.py`).
- Fara emoji in cod, decat daca se cere explicit.
- Wrappere peste unelte externe: subprocess + testare prin mock-uri; niciodata rulat binarul real
  in unit tests (exceptie: testele end-to-end dedicate, care skip-uiesc daca binarul lipseste).
- Fiecare etapa noua de pipeline urmeaza acelasi pattern ca surorile ei (`ingestion`,
  `frame_extraction`, `recommendation`, `alignment`): serviciu care scrie stage `running` ->
  `done`/`failed`, endpoint backgrounded care intoarce `202 {"status": "started"}`, pre-check 404
  sincron inainte de a programa task-ul in fundal, catch larg (`except Exception`) ca orice eroare
  sa fie inregistrata ca `failed`, nu lasata sa blocheze stage-ul pe `running`.

# Reguli de design (web/UI)

- Aplica skill-ul `frontend-design` cand lucrezi la UI.
- Stil aprobat: negru OLED (`#050506`), colturi rotunjite generos, elegant/minimalist, Apple-like,
  un singur focus pe ecran (progressive disclosure, nu dashboard dens).
- Accent verde (`#30D158`) pentru actiuni principale si stari reusite/active; portocaliu
  (`#FF9F0A`) pentru avertizare/failed/cloud; gri neutru pentru pending/complete. Tokenii sunt in
  `static/css/app.css` (`--accent-green` etc.) — nu hardcoda valori ad-hoc, foloseste tokenii.
- Frontend: HTML/CSS/JS static, fara framework, fara build step, servit prin `StaticFiles`.
- Orice string controlat de utilizator randat in DOM trece prin `escapeHtml(...)` (in `app.js`).
- Cere aprobare explicita inainte de a considera un UI final/gata de merge; un "ok" vag nu e
  sign-off pe detalii concrete (culori, tipografie, layout).

# Structura proiectului

- `app/main.py` — aplicatia FastAPI; monteaza toate router-ele API, apoi `StaticFiles` la `/`
  ca ULTIMA ruta (ordinea conteaza: rutele API trebuie inregistrate inaintea mount-ului static).
- `app/db.py` — schema SQLite (`jobs`, `job_stages`) + `connect()`.
- `app/models.py` — dataclass-urile `Job` si `JobStage`.
- `app/job_store.py` — `JobStore`: CRUD job-uri + stage-uri (SQLite).
- `app/paths.py` — helperi de cai per-job (`raw_dir`, `frames_dir`, `alignment_dir`, ...).
- `app/deps.py` — `get_job_store` (dependency FastAPI) + `run_with_new_store` (helper pentru
  task-uri de fundal, cu conexiune SQLite proprie).
- `app/ingestion.py`, `app/uploads.py` — import local + upload chunked/resumabil.
- `app/sharp_frames.py` + `app/extraction.py` — wrapper `sharp-frames` + serviciul de extragere.
- `app/video_metadata.py` (ffprobe), `app/drone_telemetry.py` (parser SRT drona),
  `app/hardware.py` (GPU/CPU/RAM) — intrari pentru motorul de recomandare.
- `app/recommendation.py` (functie pura) + `app/recommendation_service.py` (orchestrare) —
  setari RealityScan/LichtFeld + verdict local-vs-cloud.
- `app/realityscan.py` + `app/alignment_service.py` — wrapper CLI RealityScan (verificat pe
  binarul real) + serviciul de alignment.
- `app/lichtfeld.py` + `app/training_service.py` — wrapper CLI LichtFeld Studio (flag-uri
  verificate pe binarul real; comportamentul unui training real inca neverificat in aceasta
  sesiune) + serviciul de training local.
- `app/api/*.py` — router-ele HTTP (thin), unul per domeniu.
- `static/*.html` + `static/css/app.css` + `static/js/app.js` — UI-ul mobil-first.
- `tests/` — pytest; `conftest.py` are fixture-urile `job_store` si `client`. Testele
  `test_*_end_to_end.py` ruleaza binarele reale si skip-uiesc daca lipsesc.
- `run.py` — porneste serverul pe `0.0.0.0:8000`.
- `docs/superpowers/specs/` si `docs/superpowers/plans/` — spec-ul de design si planurile de
  implementare (istoricul deciziilor).

Stare de rulare (video, cadre, output, DB) traieste in `data/` (gitignored), foldere per-job;
fisierele mari nu se dubleaza — se tin referinte la cai in SQLite.

# Stadiu implementare

Facut (in `master`): fundatie (job store, ingestie, extragere cadre) · recomandare (metadata +
hardware + motor) · UI mobil + task-uri in fundal + polling · alignment RealityScan · training
LichtFeld local.

De facut: integrare RunPod (cloud GPU) ca fallback pentru training, pentru cazul in care GPU-ul
local nu indeplineste cerintele LichtFeld (vezi verdictul `use_cloud` din `app/recommendation.py`;
infra e deja construita in `app/runpod_client.py`/`app/remote_exec.py`/`app/cloud_job.py`, dar
nelegata inca de stage-ul de training). Tailscale: codul e gata (server pe `0.0.0.0`), setup-ul
efectiv pe telefon + PC ramane de facut.

# Practici de lucru (disciplina de sesiune)

- Memorie persistenta: salveaza activ feedback ("nu mai face X", "asta a mers bine, pastreaza").
- Aprobare explicita inainte de a considera ceva "gata"/"final", mai ales design.
- TodoWrite pentru orice task cu 3+ pasi, actualizat in timp real.
- Verificare reala inainte de a raporta ceva terminat: ruleaza teste, verifica output real.
- Confirma inainte de actiuni greu reversibile sau vizibile pentru altii (push, deploy, force-push).
- Nu presupune scope-ul: dupa un plan agreat, incepe implementarea doar dupa accept explicit.
- Owner-ul lucreaza des prin remote access si nu poate interveni manual daca un proces se blocheaza
  — orice apel de subprocess catre un binar extern trebuie sa fie time-bounded (vezi `timeout_seconds`
  in `realityscan.py`).

# Unelte disponibile (verifica ca sunt active in sesiunea noua)

- Plugin `superpowers`: brainstorming, TDD, systematic-debugging, writing-plans,
  subagent-driven-development, verification-before-completion.
- Plugin `frontend-design` (web/UI): skill de design, invocat la task-uri de UI.
- Plugin `chrome-devtools-mcp` (web/UI): screenshot-uri, emulare dimensiuni, Lighthouse.
- MCP `magic` (21st.dev, web/UI): cautare/generare componente UI reale.
