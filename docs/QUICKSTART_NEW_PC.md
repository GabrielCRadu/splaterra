# Quick start pe un PC nou

Acest proiect a fost mutat de pe alt PC prin copierea intregului folder (inclusiv `.git/` si
`data/`), nu prin `git clone` de pe un remote — repo-ul nu are remote configurat. Istoricul git,
job-urile existente si baza de date SQLite (`data/splatprep.db`) ar trebui sa fie deja prezente
daca folderul a fost copiat integral.

## Ce sa verifici / instalezi pe noul PC

1. **Python 3.14** — verifica `python --version`. Recreeaza venv-ul (nu se copiaza `.venv/`,
   e in `.gitignore`):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```
2. **Binare externe pe PATH** (nu vin cu venv-ul, trebuie instalate separat pe noul PC):
   - `ffmpeg` / `ffprobe`
   - `sharp-frames` (vine prin `pip install -r requirements.txt`, dar verifica ca ruleaza)
   - **RealityScan** — NU e pe PATH, se cauta prin fallback la
     `C:\Program Files\Epic Games\RealityScan_*\RealityScan.exe`. Trebuie reinstalat separat pe
     noul PC daca alignment-ul trebuie sa functioneze acolo.
3. **GPU**: daca noul PC are GPU NVIDIA (CUDA), training LichtFeld local ar putea deveni
   fezabil — pe PC-ul vechi nu mergea (GPU AMD, fara CUDA), de-aia planul era RunPod cloud. Merita
   reevaluat daca hardware-ul s-a schimbat.
4. **Tailscale**: daca vrei acces de pe telefon ca pe PC-ul vechi, Tailscale trebuie reinstalat si
   reconectat la noua masina (serverul deja asculta pe `0.0.0.0:8000`, cod-ul e gata).

## Verificare ca totul functioneaza

```
python -m pytest -q      # ar trebui sa treaca (testele *_end_to_end skip-uiesc daca lipsesc binarele)
python run.py             # porneste serverul; UI la http://127.0.0.1:8000/
```

Daca `data/` a venit cu folderul, job-urile vechi ar trebui sa apara direct in UI la pornire
(paths sunt relative fata de working directory / `SPLATPREP_DATA_DIR`, deci portabile).

## Context pentru sesiunea Claude de pe noul PC

- Citeste `CLAUDE.md` din radacina — are toate conventiile de proiect, structura, stadiul
  implementarii.
- Backlog de idei nerezolvate: `docs/TODO.md`.
- Istoricul deciziilor de design/plan: `docs/superpowers/specs/` si `docs/superpowers/plans/`.
- **Memoria persistenta a Claude Code** (`~/.claude/projects/<hash-proiect>/memory/`) traieste in
  home directory-ul userului, NU in repo — nu se muta automat odata cu folderul de proiect. Daca
  vrei sa pastrezi memoria de context/feedback acumulata, copiaz-o manual de pe PC-ul vechi:
  `C:\Users\radug\.claude\projects\c--Users-radug-Documents-Claude-Code-gaussian-splatting-app\memory\`
  catre echivalentul ei pe noul PC (calea depinde de path-ul complet al proiectului acolo). Altfel
  sesiunea Claude de pe noul PC porneste fara istoric — foloseste acest fisier + `CLAUDE.md` ca
  sa recapete contextul rapid.

## De facut, ramas neterminat (vezi si CLAUDE.md "Stadiu implementare")

- Training LichtFeld + integrare RunPod (cloud GPU) — pasul final, neinceput.
- Setup Tailscale efectiv pe telefon + noul PC (codul e gata, doar config-ul ramane).
