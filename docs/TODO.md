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
