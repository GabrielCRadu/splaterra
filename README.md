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
