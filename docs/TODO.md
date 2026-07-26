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
- [ ] La ecranul de extragere cadre: afiseaza nr. total de cadre din clip si framerate-ul clipului
      (metadata video, deja disponibila prin `app/video_metadata.py`/ffprobe).
- [ ] Metoda de extragere implicita noua: batch selection, cu aceeasi logica ca pe
      https://sharp-frames.reflct.app/ (referinta data de owner pentru comportamentul exact dorit).
- [ ] Posibilitatea de a incarca direct cadre deja extrase (skip peste extragerea automata), plus
      posibilitatea de a adauga mai mult de un clip video per job.
- [ ] Buton "advanced view" (per stage sau global pe job) ca sa vezi output-ul brut al CLI-urilor
      RealityScan si LichtFeld — logurile sunt deja salvate (`job_stages.log`), doar nu sunt
      expuse inca in UI.
- [ ] Vizualizare usage CPU / GPU / VRAM GPU / RAM al computerului in timp real, pe parcursul
      procesarii (extragere, alignment, training) — utila mai ales la training, unde owner-ul
      lucreaza des prin remote access si nu poate verifica altfel daca GPU-ul e efectiv folosit.
