# TODO

Backlog de idei, in orice ordine. Adaugat de owner, de rezolvat de Claude in sesiuni viitoare.

- [x] UI-ul de stergere a job-urilor refacut in stil iOS: press-and-hold + jiggle + drag-to-trash
      cu popup de confirmare [Delete][Cancel]. Facut in `c58aa1a`.
- [x] Animatie de loading in meniul de revizuit cadre, cat timp se incarca pozele. Grid schelet
      (shimmer) afisat imediat cat timp se incarca lista de cadre, plus shimmer per-imagine deja
      existent cat timp se incarca fiecare poza. Facut in `static/frames.html`.
- [x] La extragerea de frame-uri: optiune de a seta manual frecventa extragerii si criteriul de
      extragere, plus optiunea de a merge pe setari recomandate automat de aplicatie in functie de
      proiectul curent. Era deja implementat in `static/job.html` (panoul "Setari extragere"),
      doar nebifat in backlog.
- [x] Elimina outline-ul albastru default de focus/tap al browserului pe butoanele din UI.
      Inlocuit cu `:focus-visible` in accent verde (pastreaza accesibilitatea la navigare cu
      tastatura, elimina inelul albastru la tap/click). Facut in `static/css/app.css`.
- [x] La ecranul de extragere cadre: afiseaza nr. total de cadre din clip si framerate-ul clipului.
      Endpoint nou `GET /jobs/{id}/video-info` (estimare `round(durata * fps)`, acelasi mod de
      calcul deja folosit in `app/recommendation.py`), afisat in panoul "Setari extragere" din
      `static/job.html`. Facut in `1bddf60`.
- [x] Metoda de extragere implicita noua: batch selection, cu aceeasi logica ca pe
      https://sharp-frames.reflct.app/. Verificat direct in docs.reflct.app/capturing_gaussian_splats:
      ghidul lor pentru 3DGS recomanda exact ce foloseste deja `sharp-frames` ca default pentru
      "batched" (batch_size=5, batch_buffer=2, la >=10fps) — n-a fost nevoie de flag-uri noi, doar
      schimbat default-ul in `app/recommendation.py::recommend_extraction()` si in
      `EXTRACT_DEFAULTS` din `static/job.html`. Facut in `5954c1c`.
- [x] Posibilitatea de a incarca direct cadre deja extrase (skip peste extragerea automata).
      Cale locala catre un folder (la fel ca ingestia de video local) — sectiune noua pe pagina
      jobului, `POST /jobs/{id}/import-frames`, marcheaza stage-ul `frame_extraction` ca `done`
      direct, fara sa treaca prin `sharp-frames`. Independent de ingestia video — merge si fara
      video ingerat. Facut in `a6baba8` (si commit-urile precedente din aceeasi sesiune).
- [x] Posibilitatea de a adauga mai mult de un clip video per job. `sharp-frames` suporta nativ
      un director cu mai multe video-uri (auto-detectat) — extragerea trimite `raw_dir(job_id)`
      in loc de un fisier cand sunt 2+ clipuri ingerate, iar recomandarea insumeaza durata pe
      toate clipurile. UI nou pe pagina jobului: lista de clipuri + adaugare inca unul. Facut in
      `40953e0` (si commit-urile precedente din aceeasi sesiune).
- [x] Buton "advanced view" (per stage sau global pe job) ca sa vezi output-ul brut al CLI-urilor
      RealityScan si LichtFeld. Toggle per stage in `static/job.html`, expune `job_stages.log`
      (deja salvat, doar nu era afisat). Facut in `3c46de4`.
- [x] Vizualizare usage CPU / GPU / VRAM GPU / RAM al computerului in timp real, pe parcursul
      procesarii (extragere, alignment, training). Panou live in `static/job.html`, vizibil cat
      timp un stage ruleaza; endpoint nou `GET /hardware/usage` (`app/hardware.py`). Facut in
      `6b76428`.
- [x] ETA (timp estimat ramas) si procent de completion (text explicit, ex. "42%", nu doar latimea
      barei) pentru alignment si training, afisate langa bara de progres cat timp stage-ul ruleaza.
      Calculat client-side in `static/job.html` din `started_at` + `progress_percent` (deja expuse
      de `GET /jobs/{id}/stages`), fara modificari de backend pentru asta. Facut in `24a03af`.
- [x] Output CLI in timp real (nu doar dupa ce stage-ul se termina), ascuns dupa acelasi buton
      "advanced view" existent — momentan `job_stages.log` se populeaza abia la final.
      `run_extraction`/`run_alignment`/`run_training` accepta acum un callback `on_log`, apelat cu
      output-ul acumulat, throttled la ~1/s. `run_alignment` a necesitat portarea la acelasi model
      cu thread de citire pe care il foloseau deja extragerea si training-ul (citea stdout blocant
      abia dupa ce procesul se termina). Facut in `24a03af`.
- [x] **Import 360 (DJI Osmo 360 / DJI Avata 360), Faza 1** — video deja stitch-uit echirectangular
      (in DJI Studio, in afara aplicatiei; `.OSV` brut nu e suportat). Selector de dispozitiv per
      job la creare (`static/new-job.html`), wrapper nou `app/three_sixty_extractor.py` peste
      https://github.com/nicolasdiolez/360Extractor (reproiectie echirectangular -> pinhole + AI
      masking optional), care inlocuieste `sharp-frames` in cadrul aceleiasi etape
      `frame_extraction` (nu o etapa noua) — restul pipeline-ului (alignment/training/export) nu
      stie si nu trebuie sa stie cum au aparut cadrele. Facut in `32db26f`.
- [x] **Faza 2 pentru importul 360 (DJI Osmo 360 / Avata 360)** — verificat direct contra tool-ului
      real instalat (`--help`, `docs/SETTINGS.md`, `src/main.py`/`core/processor.py`), nu doar
      dedus:
      - Respingere `.OSV` brut la ingestie (upload si import local) — eroare clara imediata, inainte
        sa se piarda timp/spatiu incarcand un fisier nesuportat. `app/uploads.py::start_upload` +
        `app/ingestion.py::import_local_file`.
      - Suport multi-clip pentru job-uri 360 — confirmat direct in sursa tool-ului (`os.walk` pe un
        director de input, un `<stem>_processed/` per clip sub acelasi `--output` comun, nume de
        fisiere prefixate cu numele clipului sursa, deci fara coliziuni la aplatizare) ca deja
        functioneaza identic cu directorul multi-clip trimis pe calea sharp-frames — doar eliminata
        restrictia artificiala din `app/extraction.py` care forta mereu fisierul unic pentru 360.
      - Expunerea `--layout` (cub/inel/fibonacci), `--camera-count`, `--resolution` si `--nadir-mask`
        in UI (`static/job.html`) — camera-count si masca-polara/skip-up ascunse cand layout-ul e
        "cub" (tool-ul ignora camera-count pentru cub, iar masca-polara/skip-up sunt calibrate doar
        pe fetele fixe ale cubului, fara echivalent la inel/fibonacci).
      Ramase neevaluate (fara date reale de calibrare inca): `--adaptive`/`--motion-threshold`
      (keyframing bazat pe miscare), `--export-telemetry`/`--altitude-mode` (GPS/IMU din SRT-ul
      dronei), recalibrarea `timeout_seconds=3600` din `run_360_extraction`.
- [x] Output-ul CLI din "advanced view" (`static/job.html`) sa faca auto-scroll la ultimul mesaj,
      nu sa ramana la primul. `<pre class="log-output">` e recreat la fiecare render (innerHTML) -
      scroll la `scrollHeight` dupa fiecare re-render, atat la polling cat si la deschiderea toggle-ului.
- [x] Checkbox la extragerea 360 pentru a omite cadrele orientate in sus (fata "Up" a cubului,
      de obicei doar cer, inutila pentru reconstructie), prin `--active-cameras 0,1,2,3,5`
      (index 4 = Up, verificat direct contra tool-ului real, nu doar dedus din `--help`).
- [x] Reglaje tip slider pentru training (nr. de gaussieni, nr. de iteratii), cu recomandari
      afisate langa fiecare slider — `--iter`/`--max-cap` in `app/lichtfeld.py`, override din UI cu
      fallback la recomandare pentru gaussieni; 30.000 iteratii implicit (default-ul real al
      LichtFeld, confirmat din `data/training_runs.jsonl`).
- [x] Pagina de review cadre (`static/frames.html`): stergere cadre nedorite (buton pe fiecare
      thumbnail + modal de confirmare, endpoint nou `DELETE /jobs/{id}/frames/{filename}`) si
      preview mare la tinut apasat pe un cadru (peek, nu jiggle-mode — gest diferit fata de
      stergerea job-urilor din `index.html`, ca sa nu se suprapuna cele doua interactiuni).
- [x] **Scan complet (flota de subagenti, 2 etape) + reparatii** — consistenta UI/UX si bug-uri de
      cod/crash. Cele mai importante reparate:
      - Race condition reala la pornirea unei etape (extract/align/train/export/import-frames):
        doua request-uri aproape simultane puteau trece amandoua de garda 409 inainte ca vreuna sa
        scrie "running" (scris abia in background task, dupa raspunsul 202). Fix: claim atomic
        `JobStore.try_start_stage()` (un singur `INSERT ... ON CONFLICT ... WHERE status != 'running'`),
        verificat empiric sub 20 de conexiuni concurente reale (exact 1 castigator). Vezi
        `app/deps.py::claim_stage_or_409`.
      - `process.kill()` la timeout/cancel nu omora si copiii procesului (ex. ffmpeg lansat de
        sharp-frames) pe Windows — ramaneau procese orfane consumand GPU/CPU. Fix:
        `app/process_kill.py::kill_process_tree` (psutil, recursiv), verificat contra unui proces
        copil real.
      - Toate cele 4 subprocess-wrappers foloseau `text=True` fara `encoding="utf-8"` — pe Windows
        cadea pe codepage-ul consolei (cp1252), risc de `UnicodeDecodeError` silentios in thread-ul
        de citire (aceeasi clasa de bug ca `UnicodeEncodeError`-ul care a stricat setup-ul
        360Extractor). Fix: `encoding="utf-8", errors="replace"` peste tot unde exista subprocess.
      - `lichtfeld.py` putea alege un `.ply`/`.sog` vechi ramas dintr-o rulare anterioara daca
        rularea curenta nu producea niciun output nou. Fix: filtrare dupa `mtime >= start_time`.
      - Re-extragerea cadrelor dupa ce alignment/training/export rulasera deja nu invalida nimic —
        UI-ul continua sa arate "gata" pentru rezultate care nu mai corespundeau cadrelor curente,
        iar download-urile serveau splat-ul vechi. Fix: `_invalidate_downstream_results()` in
        `app/extraction.py`.
      - Frontend: butoanele nu se dezactivau cat timp POST-ul era in zbor (dublu-click = dublu
        request) si erorile (ex. 409 "deja ruleaza") erau inghitite silentios, fara niciun feedback.
        Fix: `runAction()` + mesaje reale din `detail` in `app.js`.
      - Modal de confirmare unificat cu tranzitie spring pentru cele 3 actiuni de cancel din
        `job.html` (inainte foloseau `confirm()` nativ din browser, inconsecvent cu modalul themat
        deja folosit la stergerea job-urilor/cadrelor).
      - Consistenta: `STAGE_LABELS`/`STATUS_LABELS` mutate in `app.js` (sursa unica — `index.html`
        uita "training"/"export"), `btn-export` fara gating (era mereu vizibil), buton delete cadru
        prea mic (24px -> 32px), linkuri "Inapoi la job" adaugate pe `frames.html`/`alignment.html`/
        `recommendation.html`.
      Amanate intentionat (valoare mica / risc mai mare fata de beneficiu, notate separat mai jos):
      curatenie tokeni RGB in CSS, banner pentru etape terminate in fundal cat nu sunt vizionate.
- [x] Curatenie CSS: token-uri de canale (`--accent-red-rgb`/`--accent-orange-rgb`/`--accent-green-rgb`)
      in `:root`, refolosite prin `rgba(var(--accent-*-rgb), alpha)` peste tot unde inainte tripletul
      RGB era hardcodat manual (pill-uri, glow, `.btn-danger`, `.trash-target`, `.story-banner` etc.).
      Glow-urile existente (`--accent-*-glow`) actualizate sa refoloseasca acelasi token.
- [x] `checkAutoAdvance` (`static/job.html`) acum detecteaza si tranzitia -> "done"/"failed" pentru
      *orice* etapa (nu doar cea vizionata curent), nu doar cele "running". Banner nou, verde
      (`.story-banner-done`), cu textul "X finalizat — atinge ca sa vezi" (sau "a esuat" pentru
      failed) cand o etapa din fundal se termina cat utilizatorul e pe alt slide; se sterge automat
      cand utilizatorul ajunge pe slide-ul respectiv (tap pe banner, story-bar, sau swipe), prin
      `backgroundCompletions` in `goToSlide()`.
- [x] Manifest PWA + "Add to Home Screen" — `static/manifest.json` (`display: standalone`,
      `theme_color`/`background_color` pe negrul OLED existent), iconite in 192/512/512-maskable/
      apple-touch-icon/favicon, plus meta tag-urile aferente (`apple-mobile-web-app-*`,
      `theme-color`) pe toate cele 6 pagini statice. Fara service worker deliberat —
      `NoCacheStaticFiles` din `app/main.py` forteaza deja no-cache pe tot ce serveste static ca sa
      nu se piarda modificari cat aplicatia e in dezvoltare activa, iar un service worker cu cache
      ar lupta exact impotriva politicii aia; oricum aplicatia n-are sens offline (e o telecomanda
      pentru serverul local). Facut initial cu iconite placeholder (blob-uri), inlocuite ulterior cu
      logo-ul real.
- [x] Aplicatia redenumita **Splaterra** (era "Splat Prep") — logo primit de la owner
      (`static/brand/app-logo.png` sursa patrata 4096x4096 cu chenar transparent lateral,
      `static/brand/logo-splaterra.svg` wordmark lat, ~4.3:1). Toate iconitele PWA regenerate din
      logo-ul real (crop pe bbox-ul alpha, aplatizat pe fundalul OLED ca sa nu apara colturile
      rotunjite transparente ca artefact), varianta maskable scalata la ~68% intr-un canvas
      full-bleed ca wordmark-ul (destul de lat) sa nu fie taiat de crop-ul circular Android.
      Favicon-ul (16/32px) foloseste doar globul din logo (decupat separat din `app-logo.png`),
      nu tot wordmark-ul — ilizibil ca text la mărimea aia. Wordmark-ul SVG adaugat ca banner
      (`.brand-logo`, 22px inaltime) in capul paginii, pe toate cele 6 pagini statice.
      Redenumit complet, inclusiv codul (nu doar PWA-ul, la cererea explicita ulterioara a
      owner-ului): `manifest.json`, `<title>`/`apple-mobile-web-app-title`, `FastAPI(title=...)`,
      `CLAUDE.md`, `docs/QUICKSTART_NEW_PC.md`, si variabilele de mediu `SPLATPREP_* -> SPLATERRA_*`
      in tot codul + testele + env vars reale de pe sistem (`setx`, vechile sterse). AMANAT
      deliberat, sigur — un job rula in timpul redenumirii, nu s-a atins serverul:
      - `data/splatprep.db` (numele fisierului DB din `app/db.py::db_path()`) — ar necesita oprirea
        serverului ca sa redenumesti fisierul in timp ce e deschis de conexiunea SQLite activa.
        De facut cand nu mai ruleaza niciun job.
      - Numele directorului de proiect (`gaussian-splatting-app`) — schimbare de sistem de fisiere,
        risc mult mai mare (workspace VSCode, terminale deschise) decat un simplu text/cod edit;
        nu a fost cerut explicit, ramane de decis separat daca chiar se doreste.
- [x] **Bug UX confirmat de owner**: persistenta reala pentru setarile din `static/job.html`
      (checkbox-uri extragere, slidere training, checkbox expunere/WB variabila) — coloana noua
      `job_stages.settings_json` (populata la `"running"` de `extraction.py`/`training_service.py`,
      COALESCE ca sa nu fie stearsa de update-urile de progres/log ulterioare), expusa prin
      `JobStore`/`JobStageResponse`. UI: `static/job.html` ascunde controalele editabile si afiseaza
      "ce s-a folosit efectiv" din `settings_json` odata ce stage-ul e `running`/`done` (ramane
      editabil pe `failed`, ca sa permita retry cu setari ajustate).
- [x] Sliderele de training (nr. gaussieni, iteratii, resize_factor) si checkbox-ul de expunere/WB
      variabila ascunse cat `training` e `running`/`done`; in locul lor, iteratia curenta calculata
      live din `progress_percent` + `settings_json.iterations` (nu si nr. curent de gaussieni — LichtFeld
      nu expune un numar live, doar cel final, la sfarsit — vezi `app/training_metrics.py`).
- [x] Slider si pentru `resize_factor` (auto/1/2/4/8, pasi ficsi ceruti de LichtFeld) langa cele de
      gaussieni/iteratii din `static/job.html`; override trimis prin `POST /jobs/{id}/train`, cu
      fallback la `app/recommendation.py::training_resize_factor()` cand ramane pe "Automat".
- [x] **Optiune `--eval` in UI la training** — implementat pe baza sursei reale a LichtFeld-Studio
      (`MrNeRF/LichtFeld-Studio`, nu presupuneri): `--eval` + `--test-every` in `app/lichtfeld.py`,
      rezultatele citite direct din `metrics.csv` (scris de `MetricsReporter` in `output_path`, header
      `iteration,psnr,ssim,time_per_image,num_gaussians` — mult mai robust decat regex pe log-ul CLI,
      al carui prefix exact de logging nu a fost confirmat) in `TrainingResult.eval_metrics`, persistat
      in coloana noua `job_stages.eval_metrics_json`. UI (`static/job.html`): checkbox "Masoara
      calitatea obiectiv" + camp `test_every`, afiseaza PSNR/SSIM per iteratie de evaluare odata ce
      training-ul e `done`. Notat clar in UI ca `eval_steps` (7.000 si 30.000) sunt fixe in LichtFeld,
      neconfigurabile prin CLI. **Neverificat inca pe o rulare reala cu GPU** (doar teste cu mock-uri,
      ca de obicei) — de confirmat cand owner-ul ruleaza un training cu eval activat.
- [x] **Integrare RunPod + vast.ai (cloud GPU) ca fallback pentru training** — legat de stage-ul de
      training, opt-in explicit (checkbox "Antreneaza in cloud" in `static/job.html`, implicit oprit;
      `app/recommendation.py`'s verdict `use_cloud` ramane doar un hint afisat, NU porneste automat
      un pod care costa bani). **NEVERIFICAT PE UN CONT REAL RunPod sau vast.ai** — implementat strict
      dupa documentatia publica a fiecarui provider (verificata activ, nu ghicita: `--help`-ul SDK-ului
      RunPod + fetch direct din `docs.vast.ai` pentru endpoint-urile REST reale ale vast.ai), testat
      doar cu mock-uri (`tests/test_vast_client.py`, `tests/test_cloud_job.py`,
      `tests/test_cloud_training.py`). De confirmat cand owner-ul are acces la un cont real:
      - `app/vast_client.py` (nou) — client REST direct (`requests`, nu SDK-ul oficial `vastai`) peste
        `console.vast.ai/api/v0/{bundles,asks,instances}`: cauta oferte dupa nume GPU + pret, inchiriaza
        cea mai ieftina (sau un id de oferta explicit), asteapta `actual_status == "running"`, distruge
        instanta la final. Model de inchiriere fundamental diferit de RunPod (inchiriezi o masina deja
        listata, nu un tip de GPU generic) — ascuns in spatele aceleiasi semnaturi `create_gpu_pod` ca
        `app/runpod_client.py`, documentat explicit in cod.
      - `PodConnection` mutat din `runpod_client.py` in `remote_exec.py` (tipul comun ambilor
        provideri, nu doar RunPod).
      - `app/cloud_job.py::run_cloud_job()` primeste `provider: Literal["runpod", "vastai"]`.
      - `app/cloud_training.py` (nou) — construieste comanda CLI LichtFeld pentru rulare remote
        (`app/lichtfeld.py::build_training_command()`, extras din `run_training()` ca sa fie
        reutilizabil), ruleaza prin `cloud_job.py`, citeste rezultatul (`.ply`/`.sog` + `metrics.csv`)
        din directorul descarcat local.
      - `app/training_service.py::compute_training()` — parametri noi `use_cloud`/`cloud_provider`/
        `cloud_image`/`cloud_gpu_type`; cere explicit imagine Docker + tip GPU cand `use_cloud=True`
        (validare cu stage `failed` clar, nu crash).
      **Limitari cunoscute, de rezolvat cand se verifica pe cont real**:
      - Fara progres/log live pentru rulari in cloud (`run_remote_command` e un singur apel SSH
        blocant, nu Popen+thread ca local) — bara de progres ramane indeterminata tot timpul rularii.
      - Anularea din UI nu opreste efectiv un pod remote in curs — doar marcheaza stage-ul `failed`
        local; pod-ul continua sa ruleze/factureze pana termina sau pica timeout-ul.
      - `image_name` trebuie sa fie o imagine Docker cu LichtFeld-Studio deja instalat si pe PATH —
        aplicatia nu construieste sau publica una; owner-ul trebuie sa si-o faca separat.
      - vast.ai cere cheia SSH inregistrata in cont *inainte* de a crea o instanta (spre deosebire de
        RunPod) — pas de setup manual, nedocumentat inca in `docs/QUICKSTART_NEW_PC.md`.
- [x] **Setup Tailscale efectiv** — configurat manual de owner pe telefon + PC (in afara sesiunii,
      nu era task de cod; codul era deja pregatit dinainte, server pe `0.0.0.0:8000`).
- [x] **Redenumire fisier DB** `data/splatprep.db` -> `data/splaterra.db` — facut cand niciun server/
      job nu rula (verificat: fara proces Python activ, fara stage `running` in DB inainte de
      redenumire). `app/db.py::db_path()` actualizat, fisierul real redenumit pe disc, date
      existente verificate intacte dupa (job-ul `sfinx` citit cu succes prin noul path).
- [x] **Redenumire director proiect** — decis explicit sa RAMANA `gaussian-splatting-app` (owner-ul
      a renuntat dupa ce a confirmat initial numele nou "splaterra"). Nu se mai redeschide fara o
      cerere noua explicita.
