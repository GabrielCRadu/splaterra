# Gaussian Splatting Footage Prep — Design Spec

Data: 2026-07-22

## 1. Context și scop

Aplicație personală care face streamline la preprocesarea footage-ului (video de la
cameră/telefon/dronă) pentru reconstrucție 3D Gaussian Splatting, orchestrând RealityScan
(alignment) și LichtFeld Studio (training) fără sa intervii manual la fiecare pas.

Problema pe care o rezolvă: azi, pregătirea footage-ului (extras cadre sharp, ales setări de
alignment/training potrivite hardware-ului disponibil, decis dacă e nevoie de cloud GPU) e
manuală și repetitivă. Aplicația automatizează extragerea și recomandarea, păstrând control
uman la fiecare etapă majoră, și permite pornirea/monitorizarea job-urilor de pe telefon cât
utilizatorul e plecat de acasă.

Audiență: un singur utilizator (owner-ul aplicației), rulând pe propriul PC de acasă cu GPU
NVIDIA.

## 2. Scope V1 vs. mai târziu

**V1 (acest spec):**
- Ingestie video: folder local urmărit + upload chunked/resumabil de pe telefon, fără limită
  artificială de dimensiune.
- Extragere cadre sharp (wrapper peste `sharp-frames-python`: batched selection, outlier
  removal, top-percentage selection).
- Citire metadata din footage: EXIF/container standard (ffprobe/exiftool) pentru cameră
  foto/video normală; telemetrie `.SRT` pentru dronă (altitudine, GPS, gimbal).
- Detectare hardware local: GPU (model, VRAM, compute capability via `nvidia-smi`/`pynvml`),
  CPU/RAM (`psutil`).
- Motor de recomandare: produce setări sugerate pentru RealityScan (component settings,
  downscale, rolling-shutter correction) și LichtFeld (max gaussians, bf16, VRAM auto-fit),
  plus verdict local-vs-cloud pe baza VRAM necesar estimat vs. disponibil.
- Orchestrare RealityScan headless/CLI local, cu setările alese (editabile înainte de start).
- Orchestrare LichtFeld: headless local dacă hardware-ul ajunge, altfel upload dataset +
  provisioning automat de instanță GPU pe RunPod, training, download rezultat, teardown automat
  al instanței la final.
- UI web mobil-first, minimalist, acces remote prin Tailscale, progres live și notificări.
- Fiecare etapă majoră (extragere → alignment → training) pornită manual din UI, cu progres
  vizibil live; nu există pipeline complet automat fără confirmare.

**Explicit în afara scope-ului V1 (fazele următoare):**
- Alți provideri de cloud GPU în afară de RunPod.
- Editare avansată a norului de puncte / curățare manuală dincolo de accept-reject pe cadre.
- Suport pentru alte trainere Gaussian Splatting în afară de LichtFeld Studio.
- Pipeline complet automat, fără confirmare umană între etape.
- Animații UI (adăugate după ce funcționalitatea de bază e stabilă).

## 3. Arhitectură generală

```
[Telefon: browser]
      │  (Tailscale, retea privata)
      ▼
[PC acasă — serviciu local Python/FastAPI]
  ├─ Web UI (mobile-first, servit local, aceeași aplicație)
  ├─ Modul ingestie video (folder local urmărit + upload chunked de pe telefon)
  ├─ Modul extragere cadre sharp (wrapper peste sharp-frames-python)
  ├─ Modul metadata cameră (ffprobe/exiftool + parser SRT dronă)
  ├─ Modul detectare hardware (nvidia-smi/pynvml + psutil)
  ├─ Motor de recomandare setări (RealityScan + LichtFeld)
  ├─ Orchestrator RealityScan (CLI/headless local)
  ├─ Orchestrator LichtFeld (headless local SAU provisioning + run pe RunPod)
  └─ Job store (SQLite) + progres/log-uri live (WebSocket/SSE către UI)
```

Fiecare componentă are o responsabilitate unică și comunică prin interfețe simple (funcții
Python / apeluri HTTP interne), astfel încât orchestratoarele RealityScan și LichtFeld pot fi
testate independent de restul, prin mock-uri de subprocess.

## 4. Fluxul pipeline-ului

**Etapa 0 — Ingestie video**
- Local: fișiere adăugate într-un folder urmărit, sau selectate din UI.
- Telefon: upload chunked/resumabil prin UI, rezistent la întreruperi de rețea; limitat doar
  de spațiul pe disc, nu de o limită artificială a aplicației.

**Etapa 1 — Extragere cadre sharp**
- Rulează wrapper-ul peste `sharp-frames-python` cu metodele de selecție existente (batched /
  outlier removal / top-percentage), configurabile din UI.
- Rezultat: set de cadre + grid de preview în UI; utilizatorul poate exclude manual cadre
  proaste (accept/reject) înainte de a continua.

**Etapa 2 — Citire metadata + detectare hardware**
- Din video: focal length, dimensiune senzor, shutter speed, ISO; dacă e dronă, telemetrie SRT
  (altitudine, GPS, gimbal) — folosită pentru rolling-shutter correction și scale.
- Din PC: GPU (model, VRAM, compute capability), CPU, RAM.
- Motorul de recomandare produce: parametri sugerați RealityScan, parametri sugerați LichtFeld
  (inclusiv max gaussians limitat de VRAM/RAM disponibil), și verdictul local-vs-cloud.

**Etapa 3 — Alignment (RealityScan)**
- Utilizatorul apasă "Start alignment" în UI (setările sunt editabile înainte de pornire).
- Serviciul rulează RealityScan headless/CLI local.
- Progres live + log-uri în UI; rezultat = camere aliniate + nor de puncte.

**Etapa 4 — Training (LichtFeld)**
- Dacă hardware-ul local ajunge (conform verdictului din Etapa 2): rulează headless local.
- Dacă nu: upload dataset către RunPod (network volume), provisioning instanță cu imagine
  Docker pregătită (LichtFeld + CUDA), rulează training headless, descarcă rezultatul, oprește
  automat instanța la final (fără cost de timp mort).
- Progres live + notificare push/UI când e gata.

**Gestionare erori**: fiecare etapă e izolată — dacă RealityScan sau LichtFeld eșuează (crash,
out-of-memory), job-ul rămâne în stare `failed` cu log-ul complet vizibil în UI; nu blochează
restul aplicației. Utilizatorul poate relua doar etapa eșuată, fără să repete etapele anterioare
deja reușite (rezultatele intermediare persistă pe disc + referințe în SQLite).

## 5. UI — direcție vizuală

Aprobată vizual prin mockup (vezi `.superpowers/brainstorm/` din sesiunea de brainstorming):

- **Fundal**: negru OLED (`#000000`/`#050506`), nu gri închis.
- **Formă**: colțuri rotunjite generos (16–24px), carduri elevate subtil (`rgba(255,255,255,0.04-0.06)`
  pe negru, hairline-uri fine `rgba(255,255,255,0.06-0.08)`).
- **Tipografie**: stack system (`-apple-system`/`SF Pro`/`Segoe UI`), greutăți 600-680 pe
  titluri, letter-spacing negativ ușor pe headinguri.
- **Accent**: verde (`#30D158`, stil iOS) pentru acțiuni principale și stări reușite/active —
  confirmat ca preferat. Portocaliu (`#FF9F0A`) pentru stări de avertizare/cloud GPU activ,
  gri neutru pentru stări neutre/complete.
- **Densitate**: minimalistă, un singur focus pe ecran; detalii avansate (log-uri, setări)
  ascunse în spatele unui tap. Progressive disclosure, nu dashboard dens.
- **Animații**: se adaugă la final, după ce funcționalitatea de bază e stabilă — nu fac parte
  din V1 inițial.

Ecrane de referință mockup-uite: listă job-uri (Acasă), etapă activă cu progres circular +
statistici hardware live, revizuire grid de cadre extrase cu accept/reject.

## 6. Stack tehnic & persistență date

- **Backend**: Python + FastAPI; servește și web UI-ul (build static).
- **Stare job-uri**: SQLite local (un singur fișier) — fiecare job, etapele lui și log-urile
  persistă acolo, ca serviciul să reziste la restart fără să piardă progresul.
- **Fișiere**: video/cadre/output rămân pe disc local, în foldere per-job; nicio duplicare
  inutilă de gigabytes (referințe la path-uri).
- **Comunicare live**: WebSocket/SSE pentru progres și log-uri către UI.

## 7. Acces remote

- **Tailscale**: serviciul local pornește un web server accesibil prin rețeaua privată
  Tailscale; utilizatorul instalează Tailscale pe PC și pe telefon (cont gratuit personal).
  Trafic criptat end-to-end, fără porturi expuse pe internetul public.

## 8. Cloud GPU — RunPod

- Provider ales: **RunPod** (API solid, billing per-secundă, imagini Docker custom, network
  volumes pentru fișiere mari).
- Flux: dacă verdictul Etapei 2 e "cloud necesar", aplicația face upload al dataset-ului
  (cadre + alignment din RealityScan) către un network volume RunPod, pornește o instanță cu
  imaginea Docker pregătită (LichtFeld + CUDA), rulează training-ul headless, descarcă
  rezultatul final, și oprește automat instanța — nu se plătește timp mort.
- Alți provideri (Vast.ai, Lambda Labs) rămân explicit în afara scope-ului V1.

## 9. Testare

- Motorul de recomandare (metadata + hardware → setări) testat cu cazuri cunoscute de
  input/output (unit tests).
- Extragerea de cadre testată pe clipuri video mici de test, verificând numărul și calitatea
  cadrelor selectate.
- Orchestrarea RealityScan/LichtFeld testată prin mock-uri de subprocess (nu există CLI de test
  oficial pentru unelte proprietare) — verifică parametrii trimiși și gestionarea corectă a
  stărilor `running`/`failed`/`done`.
- Testare manuală end-to-end pe un caz real (footage real, alignment + training complet, local
  și pe RunPod) înainte de a considera V1 gata.
