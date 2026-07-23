# Splat Prep

Aplicatie de preprocesare footage pentru Gaussian Splatting: extragere cadre sharp,
detectare hardware, recomandare setari RealityScan/LichtFeld.

## Rulare locala

```bash
python -m pip install -r requirements.txt
python run.py
```

Serverul asculta pe `0.0.0.0:8000` — deschide `http://127.0.0.1:8000/` in browser pe acelasi PC.

## Acces remote (Tailscale)

Ca sa accesezi aplicatia de pe telefon, de oriunde, prin reteaua ta privata Tailscale:

1. Instaleaza [Tailscale](https://tailscale.com/download) pe PC-ul care ruleaza serverul si pe telefon, folosind acelasi cont.
2. Pe PC, afla adresa Tailscale a masinii: `tailscale ip -4` (sau din aplicatia Tailscale).
3. Cu `python run.py` pornit (asculta pe `0.0.0.0`, deci e deja accesibil pe orice interfata, inclusiv cea Tailscale), deschide pe telefon, in aplicatia Tailscale activa: `http://<adresa-tailscale>:8000/`.
4. Nu e nevoie de nicio configurare suplimentara in cod — Tailscale ruteaza direct catre portul pe care asculta serverul.

**Atentie:** serverul asculta pe `0.0.0.0`, deci e accesibil oricui e pe aceeasi retea locala (WiFi), nu doar prin Tailscale — aplicatia nu are autentificare. Ruleaz-o doar pe retele de incredere (acasa, nu WiFi public).

## Cloud GPU (RunPod)

Training pe LichtFeld cere un GPU NVIDIA — daca PC-ul tau nu are unul (sau nu e suficient de
puternic), aplicatia poate rula training-ul pe un pod GPU temporar in cloud, prin RunPod.

Pasi de configurare (o singura data):

1. Creeaza cont pe [runpod.io](https://www.runpod.io) si genereaza un API key din
   Settings -> API Keys.
2. Seteaza-l ca variabila de mediu: `RUNPOD_API_KEY=<cheia-ta>` (nu il pune niciodata in cod
   sau in git).
3. Adauga cheia ta publica SSH (`~/.ssh/id_rsa.pub` sau echivalent) in contul RunPod, la
   Settings -> SSH Public Keys — fara asta, aplicatia nu se poate conecta la pod-urile create.
4. Daca folosesti alta cale pentru cheia privata SSH decat cea default (`~/.ssh/id_rsa`),
   seteaza `SPLATPREP_SSH_KEY_PATH` catre calea corecta.

Costul e per-ora, doar cat timp ruleaza pod-ul — aplicatia opreste automat pod-ul la finalul
fiecarui job (sau la orice eroare), ca sa nu platesti timp mort.
