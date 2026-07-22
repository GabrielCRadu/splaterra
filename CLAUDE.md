# Context proiect

[Ce e proiectul, in 3-5 propozitii. Pentru cine e (audienta). Ce actiune concreta vrei de la
utilizatorul final / ce problema rezolvi.]

# Comenzi

[Comenzi REALE, verificate, nu presupuse: dev/build/lint/test/deploy. Daca o comanda de deploy
nu exista inca, scrie explicit "nu presupune o comanda pana nu e configurata explicit".]

# Conventii de cod

- Fara comentarii inutile: doar cand explica un WHY neevident (constrangere ascunsa, bug
  workaround), niciodata un WHAT care se vede din cod.
- [Daca scrii continut/documentatie intr-o limba cu diacritice, dar codul e in engleza: fara
  diacritice in cod/commit-uri/nume de fisiere, dar OBLIGATORIU diacritice corecte in continutul
  vizibil utilizatorului final (texte, UI, mesaje). Regula se aplica strict codului, nu continutului.]
- Fara emoji in cod, decat daca se cere explicit.
- [Alte conventii specifice: naming, structura de foldere, limbaj/framework.]

# Reguli de design (doar web/UI, sterge sectiunea daca nu se aplica)

- Aplica skill-ul `frontend-design` cand lucrezi la UI.
- Evita cliseele vizuale AI: crem+serif+terracota, negru+neon, layout de ziar cu linii subtiri.
- Construieste un token system (culori, spatiere, tipografie, radius, umbre) inainte de componente
  individuale, nu hardcoda valori ad-hoc.
- Pentru componente noi: cauta intai surse externe/librarii de componente reale (ex. 21st.dev prin
  MCP-ul `magic`), apoi alte referinte de design ca inspiratie, abia apoi construiesti de la zero.
- Daca exista o paleta stricta: accentele de culoare NU se adauga pe cont propriu, doar cand
  userul cere explicit pe ce anume.
- **Cere aprobare explicita inainte de a considera un UI final/gata de merge.** Nu trata un "ok"
  sau "hai cu asta" vag ca sign-off pe detalii concrete (culori, tipografie, layout) - intreaba direct.

# Reguli de responsive design (doar web/UI, sterge sectiunea daca nu se aplica)

Obligatoriu la orice task de UI, oricat de mic pare:
1. `clamp(min, preferred, max)` pentru tipografie/spatiere, nu salturi fixe intre breakpoint-uri.
2. Container queries in loc de media queries acolo unde e posibil.
3. Grid-uri fluide: `repeat(auto-fit, minmax(XXXpx, 1fr))`, nu numar fix de coloane.
4. Unitati relative: rem pentru tipografie, % / fr pentru latimi.
5. `dvh` in loc de `vh` pentru sectiuni full-height pe mobil.
6. Imagini: `max-width: 100%`, `aspect-ratio` explicit, niciodata dimensiuni fixe care distorsioneaza.
7. `flex-wrap` implicit pe orice flex container cu continut variabil.
8. Zero overflow orizontal, la nicio latime.
9. Breakpoint-urile clasice (sm/md/lg) doar pentru schimbari majore de layout, nu ca metoda
   principala de responsive.

Verificare completa (doar cand se cere explicit sau inainte de sign-off final, nu la fiecare
iteratie mica): screenshot + test la 320x568, 375x667, 390x844, 667x375, 768x1024, 1024x768,
1366x768, 1920x1080, 2560x1080, 400x900, 800x400. La fiecare: overflow orizontal, text taiat,
imagini distorsionate, tap targets sub 44px, spatii goale nejustificate. Apoi Lighthouse
(performanta, accesibilitate, best practices, SEO).

# Structura proiectului

[Harta scurta a folderelor/fisierelor importante si ce face fiecare.]

# Practici de lucru (disciplina de sesiune)

- **Memorie persistenta**: salveaza activ feedback ("nu mai face X", "asta a mers bine, pastreaza")
  ca sa nu se repete aceleasi corectii intre sesiuni.
- **Aprobare explicita** inainte de a considera ceva "gata"/"final", mai ales design sau
  implementare dupa un plan agreat - un "ok" vag nu inseamna sign-off pe detalii concrete.
- **TodoWrite** pentru orice task cu 3+ pasi, actualizat in timp real (nu la final, in bloc).
- **Verificare reala inainte de a raporta ceva ca terminat**: ruleaza type-check/lint/teste,
  fa screenshot-uri, ruleaza audituri de performanta. Nu presupune ca a mers.
- **Confirma inainte de actiuni greu reversibile** sau vizibile pentru altii (push, deploy,
  stergere de continut live, force-push) - chiar daca ai autonomie generala, verifica la
  devieri de la planul aprobat.
- **Nu presupune scope-ul**: daca userul aproba un plan/spec, nu incepe implementarea pana nu
  primesti un accept explicit dat DUPA ce planul exista.

# Unelte disponibile (verifica ca sunt active in sesiunea noua)

- Plugin `superpowers`: brainstorming inainte de implementare, TDD, systematic-debugging,
  verification-before-completion, writing-plans.
- Plugin `frontend-design` (doar web/UI): skill de design, invocat explicit la task-uri de UI.
- Plugin `chrome-devtools-mcp` (doar web/UI): screenshot-uri, emulare de dimensiuni, Lighthouse,
  performance trace.
- MCP `magic` (21st.dev, doar web/UI): cautare/generare de componente UI reale.
