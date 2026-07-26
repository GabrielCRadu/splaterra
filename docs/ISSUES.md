# Issues

Probleme cunoscute care trebuie investigate/rezolvate, descoperite in timpul dezvoltarii sau
testarii. Spre deosebire de `docs/TODO.md` (idei noi de facut), aici sunt lucruri care nu
functioneaza corect sau al caror comportament e neverificat.

- [ ] `tests/test_realityscan.py::test_run_alignment_reports_progress_from_progress_file` pica
      intermitent cand ruleaza suita completa de teste (`python -m pytest -q`), dar trece mereu
      cand ruleaza izolat. Aparut de 3 ori in aceeasi sesiune (2026-07-26). Nu pare noroc curat —
      arata a test-pollution/ordine intre teste (state global sau mock care scapa intre module),
      nu un bug de runtime in `app/realityscan.py`. De investigat separat.
- [ ] Training LichtFeld (`app/lichtfeld.py`): doua ipoteze neverificate contra unui training real
      (documentate in cod si in planul de implementare, dar inca netestate pe teren):
      1. Formatul exact al progresului pe stdout in timpul training-ului — parsare best-effort cu
         mai multe pattern-uri incercate, fallback la bara indeterminata daca niciunul nu se
         potriveste.
      2. Conventia exacta de denumire/locatie a fisierului de iesire (.ply/.sog) — rezolvat prin
         glob recursiv dupa cel mai recent fisier `.ply`/`.sog`, nu cale hardcodata.
      Se verifica rulland `tests/test_training_end_to_end.py` cu `SPLATPREP_E2E_ALIGNMENT_DIR` +
      `SPLATPREP_RUN_REAL_LICHTFELD_TEST=1` contra unui job cu alignment real complet, sau direct
      din UI apasand "Porneste antrenarea" pe un job real.
