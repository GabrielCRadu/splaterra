# Issues

Probleme cunoscute care trebuie investigate/rezolvate, descoperite in timpul dezvoltarii sau
testarii. Spre deosebire de `docs/TODO.md` (idei noi de facut), aici sunt lucruri care nu
functioneaza corect sau al caror comportament e neverificat.

- [x] `tests/test_realityscan.py::test_run_alignment_reports_progress_from_progress_file` pica
      intermitent cand ruleaza suita completa de teste, dar trecea mereu izolat — gasit si reparat
      2026-07-26. Cauza: `test_run_extraction_raises_on_timeout` (`test_sharp_frames.py`) si
      `test_run_training_raises_on_timeout` (`test_lichtfeld.py`) folosesc fiecare un `Popen.stdout`
      fals care nu semnaleaza niciodata EOF — thread-ul real `_reader_thread` (`daemon=True`) pornit
      de `run_extraction`/`run_training` ramane deci sa ruleze la infinit, apeland `time.sleep(0.01)`
      prin `import time as real_time` **re-executat la fiecare citire**. Cum `time` e un modul
      singleton la nivel de proces, acel `real_time` e literal acelasi obiect ca `realityscan.time`
      — cand testul de progres monkeypatch-uieste `realityscan.time.sleep`, thread-ul scapat dintr-un
      test anterior (daca inca traieste) apela si el `fake_sleep`, incrementand
      `sleep_calls["count"]` dintr-un thread necontrolat si stricand secventa exacta pe care testul
      o astepta (raporta 90.0 in loc de 30.0 ca prim procent). Fix: capturata o referinta directa
      `_REAL_SLEEP = time.sleep` la nivel de modul, inainte sa poata rula vreun monkeypatch — imun
      structural la orice patch ulterior, indiferent de timing. Verificat cu 3 rulari complete
      consecutive ale suitei, toate curate (246 passed, 0 failed).
- [x] Training LichtFeld (`app/lichtfeld.py`): cele doua ipoteze erau neverificate contra unui
      training real — confirmate 2026-07-26 pe un job real (`MasinaPoarta`, 30000 iteratii, 4.8M
      gaussieni, 58 minute, finalizat cu `status: done`, fara erori):
      1. **Formatul progresului pe stdout**: NU e stil tqdm (`NN%|`), ci
         `Training [====>] 98% [57m:14s<01m:10s] 29400/30000 | Loss: 0.0298 | Splats: 4800000`.
         Fallback-ul pe fractia `N/M` (`_FRACTION_RE`) a prins corect progresul pe tot parcursul
         antrenarii, pana la 99.67% (ultima linie citita inainte de mesajele de finalizare) — nu
         a fost nevoie de niciun cod nou, parsarea best-effort deja acoperea acest caz.
      2. **Conventia fisierului de iesire**: LichtFeld scrie splat-ul final direct in
         `<output_dir>/splat_<iteratii>.ply` (ex. `training/splat_30000.ply`, ~1.19 GB pentru 4.8M
         gaussieni), separat de checkpoint-urile de resume salvate in
         `<output_dir>/checkpoints/checkpoint_<iter>.resume`. Glob-ul recursiv dupa `.ply`/`.sog`
         a gasit fisierul corect fara ambiguitate (checkpoint-urile `.resume` nu se potrivesc cu
         patternul, deci nu interfereaza).
      Nu mai e nevoie de investigatie suplimentara — comportamentul e confirmat pe teren.
