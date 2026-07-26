# Issues

Probleme cunoscute care trebuie investigate/rezolvate, descoperite in timpul dezvoltarii sau
testarii. Spre deosebire de `docs/TODO.md` (idei noi de facut), aici sunt lucruri care nu
functioneaza corect sau al caror comportament e neverificat.

- [ ] `tests/test_realityscan.py::test_run_alignment_reports_progress_from_progress_file` pica
      intermitent cand ruleaza suita completa de teste (`python -m pytest -q`), dar trece mereu
      cand ruleaza izolat. Aparut de 3 ori in aceeasi sesiune (2026-07-26). Nu pare noroc curat —
      arata a test-pollution/ordine intre teste (state global sau mock care scapa intre module),
      nu un bug de runtime in `app/realityscan.py`. De investigat separat.
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
