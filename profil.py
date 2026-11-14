"""Profilkontroll: bedömer varje kandidats egna publikationer mot temana.

Sökningen i rekrytera.py tar bara fram namn. Här hämtas varje kandidats
publikationer från de senaste åren, och för varje tema räknas hur många av
dem som handlar om temat, hur nya de är och vilken roll kandidaten hade.

Likheten mellan ett tema och en publikation räknas lokalt på datorn, antingen
med en språkmodell (om sentence-transformers och modellen finns installerade)
eller med ordjämförelse (TF-IDF). Språkmodellen förstår synonymer och är
träffsäkrare, ordjämförelsen är snabbare och kräver inget extra.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from openalex import MANGA_FORFATTARE, OpenAlex

AR = dt.date.today().year

# Engelsk modell, liten och snabb. Publikationerna i OpenAlex är nästan bara på
# engelska och temana ska skrivas på engelska. Byt till
# "paraphrase-multilingual-MiniLM-L12-v2" om temana måste vara på svenska.
SPRAKMODELL = "all-MiniLM-L6-v2"
TEXTLANGD = 1000        # tecken av titel + abstract som jämförs med temat

# Relevans är en skala 0–1 per tema: 0 = en typisk publikation bland kandidaterna,
# 1 = temats bästa sökträffar (se kalibrera). Gränserna är satta efter granskning
# av riktiga körningar: under 0,5 handlar verken oftast om något annat inom samma
# breda fält, över 0,65 nästan alltid om temat.
RELEVANT = 0.65         # från denna relevans räknas ett verk som relevant för temat
GOLV = 0.5              # under detta bidrar verket inte alls; fullt bidrag från 0,8
MATTNAD = 7.0           # fem fullt relevanta verk ger temastyrka 0,5; tio 0,76; tjugo 0,94
AKTUELLT_AR = 5         # verk från de senaste fem åren väger fullt, äldre 0,75

# Undantag: ett verk räknas till ett undantaget område när det liknar
# undantagets exempelpublikationer minst 10 % mer än något temas exempel och
# når minst 70 % av den likhet exemplen själva har med varandra.
UNDANTAGSMARGINAL = 1.1
UNDANTAGSNIVA = 0.7

_HOPPA = re.compile(
    r"^\s*(?:additional file|supplementa(?:ry|l) (?:material|file|data|table|figure|information|appendix)|"
    r"(?:author |publisher )?(?:correction|corrigendum|erratum|retraction)\b\s*(?:to\b|:|notice)|"
    r"author response|peer review|decision letter|table of contents|front matter|editorial board)",
    re.IGNORECASE)
_LAGVIKT = re.compile(
    r"\b(?:protocol|editorial|commentary|research priorit\w*|priority[- ]setting|research agenda|"
    r"roadmap|call for action|consensus statement|delphi|position (?:paper|statement))\b",
    re.IGNORECASE)


# ---------------------------------------------------------------- vikter
def titelvikt(titel: str) -> float:
    """0 för sådant som inte är forskning (bilagor, rättelser), 0,5 för protokoll,
    ledare och prioriteringsartiklar, annars 1."""
    if not titel or _HOPPA.search(titel):
        return 0.0
    return 0.5 if _LAGVIKT.search(titel) else 1.0


def rollvikt(position: str | None, antal_forfattare: int, korresponderande: bool = False) -> float:
    """Första-, sista- och korresponderande författare väger fullt. En
    mittenförfattare väger 0,6 i en liten grupp och mindre ju fler författarna är."""
    if antal_forfattare <= 1 or position in ("first", "last") or korresponderande:
        return 1.0
    return 0.6 * min(1.0, 4 / antal_forfattare)


def aktualitet(ar: int | None) -> float:
    return 1.0 if ar and ar >= AR - AKTUELLT_AR else 0.75


def norm_titel(titel: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (titel or "").lower()).strip()


def verktext(verk: dict) -> str:
    return f"{verk['t']}. {verk['a']}".strip()[:TEXTLANGD]


# ---------------------------------------------------------------- hämtning
def parallellt(funk, ids: list[str], text: str = "", framsteg=None, tradar: int = 4) -> dict:
    """Kör funk(id) för varje id, fyra i taget. Ger {id: svar}, med None för
    den som inte gick att hämta."""
    stopp = threading.Event()

    def en(x: str):
        if stopp.is_set():
            return x, None
        try:
            return x, funk(x)
        except (RuntimeError, requests.RequestException):
            return x, None
        except BaseException:
            stopp.set()     # t.ex. slut på budget: avbryt de återstående direkt
            raise

    ut: dict = {}
    with ThreadPoolExecutor(max_workers=tradar) as pool:
        for nr, (x, svar) in enumerate(pool.map(en, ids), 1):
            ut[x] = svar
            if framsteg and text and (nr % 10 == 0 or nr == len(ids)):
                framsteg(f"{text} ({nr}/{len(ids)}) …", nr / len(ids))
    return ut


def hamta_verk(oa: OpenAlex, ids: list[str], fran_ar: int, framsteg=None) -> dict:
    """Verken för varje forskare: {id: {"totalt", "verk"}}."""
    return parallellt(lambda aid: oa.verk_kompakt(aid, fran_ar), ids,
                      "Hämtar publikationer för kandidaterna", framsteg)


def rensa_verk(verk: list[dict]) -> list[dict]:
    """Tar bort bilagor och rättelser och slår ihop poster med samma titel
    (t.ex. preprint och tryckt artikel)."""
    plats: dict[str, int] = {}
    ut: list[dict] = []
    for v in verk:
        if titelvikt(v["t"]) == 0:
            continue
        nyckel = norm_titel(v["t"]) or v["id"]
        if nyckel in plats:
            j = plats[nyckel]
            if len(v["a"]) > len(ut[j]["a"]):
                ut[j] = {**ut[j], "a": v["a"]}
            continue
        plats[nyckel] = len(ut)
        ut.append(v)
    return ut


# ---------------------------------------------------------------- likhet
_MODELLER: dict[str, object] = {}


def ladda_modell(namn: str = SPRAKMODELL, nedladdning: bool = False):
    """Laddar språkmodellen. Utan `nedladdning` används den bara om den redan
    finns på datorn. Ger None om den inte går att ladda."""
    if namn in _MODELLER:
        return _MODELLER[namn]
    try:
        from sentence_transformers import SentenceTransformer
        modell = SentenceTransformer(namn, local_files_only=not nedladdning)
    except Exception:       # paketet saknas, modellen är inte nedladdad, inget nät …
        return None
    modell.max_seq_length = 200
    _MODELLER[namn] = modell
    return modell


def valj_likhet(val: str = "auto"):
    """Ger ("modell", modellen) eller ("tfidf", None)."""
    if val == "tfidf":
        return "tfidf", None
    modell = ladda_modell(nedladdning=(val == "modell"))
    if modell is not None:
        return "modell", modell
    if val == "modell":
        raise SystemExit("Språkmodellen kunde inte laddas. Kör: pip install sentence-transformers "
                         "(modellen laddas ner första gången, cirka 90 MB).")
    return "tfidf", None


def _koda_verk(modell, verk_per_forfattare: dict, cache_dir: Path | None, framsteg=None) -> np.ndarray:
    """Vektorer för alla verk, i samma ordning som verk_per_forfattare. Vektorerna
    sparas per forskare i cachen, så att en ny körning med andra teman går fort."""
    slug = re.sub(r"[^a-z0-9]+", "-", SPRAKMODELL.lower()).strip("-")
    dim = modell.get_sentence_embedding_dimension()

    def fil(aid: str) -> Path | None:
        return cache_dir / f"emb-{slug}-t{TEXTLANGD}-{aid}.npz" if cache_dir else None

    delar: dict[str, np.ndarray] = {}
    saknas: list[tuple[str, int, str]] = []
    for aid, verk in verk_per_forfattare.items():
        E = np.zeros((len(verk), dim), dtype=np.float32)
        gamla: dict = {}
        f = fil(aid)
        if f and f.exists():
            try:
                with np.load(f, allow_pickle=False) as z:
                    gamla = dict(zip(z["ids"].tolist(), z["E"]))
            except Exception:
                gamla = {}
        for j, v in enumerate(verk):
            if v["id"] in gamla:
                E[j] = gamla[v["id"]]
            else:
                saknas.append((aid, j, verktext(v)))
        delar[aid] = E

    for start in range(0, len(saknas), 256):
        bit = saknas[start:start + 256]
        vek = modell.encode([t for _, _, t in bit], normalize_embeddings=True, batch_size=32,
                            show_progress_bar=False)
        for (aid, j, _), e in zip(bit, vek):
            delar[aid][j] = e
        if framsteg:
            klart = min(start + 256, len(saknas))
            framsteg(f"Språkmodellen läser publikationerna ({klart}/{len(saknas)}) …", klart / len(saknas))
    if cache_dir:
        for aid in {a for a, _, _ in saknas}:
            verk = verk_per_forfattare[aid]
            np.savez_compressed(fil(aid), ids=np.array([v["id"] for v in verk]),
                                E=delar[aid].astype(np.float16))
    if not delar:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack(list(delar.values()))


def _enhet(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _exempelniva(V, summa: np.ndarray) -> float | None:
    """Hur lika exemplen är varandra: medianen av varje exempels likhet med
    medelvektorn för de ÖVRIGA exemplen. Det egna exemplet räknas bort, annars
    överskattas likheten (mest i ordjämförelsen). V har normerade rader."""
    if V.shape[0] < 2:
        return None
    med_summan = np.asarray(V @ summa).ravel()
    ovriga_norm = np.sqrt(np.maximum(float(summa @ summa) - 2 * med_summan + 1.0, 1e-12))
    return float(np.median((med_summan - 1.0) / ovriga_norm))


def likhet(fragor: list[str], verk_per_forfattare: dict, exempel: list[list[str]], val: str = "auto",
           cache_dir: Path | None = None, framsteg=None) -> dict:
    """Likhet mellan varje fråga (tema eller undantag) och varje verk.

    `exempel[q]` är texterna för de bästa sökträffarna på fråga q, alltså
    publikationer som säkert handlar om den. Ger en dict med:
      S         – likhet mellan frågans text och varje verk (en rad per fråga, en
                  kolumn per verk i samma ordning som verk_per_forfattare)
      P         – P[q] är samma likhet för frågans exempel (till kalibrera)
      prototyp  – likhet mellan frågans exempel (deras medelvektor) och varje verk
      exempelniva – hur lika exemplen är varandra (se _exempelniva), None utan exempel
      metod     – vilken jämförelse som användes
    """
    slag, modell = valj_likhet(val)
    Q_antal = len(fragor)
    if slag == "modell":
        if framsteg:
            framsteg("Jämför publikationerna med temana (språkmodell) …", None)
        E = _koda_verk(modell, verk_per_forfattare, cache_dir, framsteg)
        Q = modell.encode(fragor, normalize_embeddings=True, show_progress_bar=False)
        S = Q @ E.T
        P, niva = [], []
        prototyp = np.zeros((Q_antal, E.shape[0]))
        for q, ex in enumerate(exempel):
            if not ex:
                P.append(np.zeros(0))
                niva.append(None)
                continue
            V = modell.encode(ex, normalize_embeddings=True, show_progress_bar=False)
            summa = V.sum(axis=0)
            P.append(V @ Q[q])
            niva.append(_exempelniva(V, summa))
            prototyp[q] = E @ _enhet(summa)
        return {"S": S, "P": P, "prototyp": prototyp, "exempelniva": niva,
                "metod": f"språkmodell ({SPRAKMODELL})"}

    from sklearn.feature_extraction.text import TfidfVectorizer

    if framsteg:
        framsteg("Jämför publikationerna med temana (ordjämförelse) …", None)
    texter = [verktext(v) for verk in verk_per_forfattare.values() for v in verk]
    platta = [t for ex in exempel for t in ex]
    stor = len(texter) + len(platta) >= 50
    vek = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2),
                          min_df=2 if stor else 1, max_df=0.7 if stor else 1.0)
    X = vek.fit_transform(texter + platta + fragor)
    n, m = len(texter), len(platta)
    W, Pm, Q = X[:n], X[n:n + m], X[n + m:]
    S = (Q @ W.T).toarray() if n else np.zeros((Q_antal, 0))
    P, niva, start = [], [], 0
    prototyp = np.zeros((Q_antal, n))
    for q, ex in enumerate(exempel):
        if not ex:
            P.append(np.zeros(0))
            niva.append(None)
            continue
        V = Pm[start:start + len(ex)]
        start += len(ex)
        summa = np.asarray(V.sum(axis=0)).ravel()
        P.append((V @ Q[q].T).toarray().ravel())
        niva.append(_exempelniva(V, summa))
        if n:
            prototyp[q] = W @ _enhet(summa)
    return {"S": S, "P": P, "prototyp": prototyp, "exempelniva": niva,
            "metod": "ordjämförelse (TF-IDF)"}


def undantagsverk(prototyp: np.ndarray, exempelniva: list, antal_teman: int) -> np.ndarray:
    """Vilka verk som hör till ett undantaget område: en rad per undantag, en
    kolumn per verk (sant/falskt).

    Ett verk räknas dit när det liknar undantagets exempelpublikationer tydligt
    mer än det liknar något temas exempel (UNDANTAGSMARGINAL) och dessutom
    ligger nära dem (UNDANTAGSNIVA av exemplens egen spridning). Att jämföra
    med exempelpublikationer är träffsäkrare än att jämföra med undantagets
    beskrivning, som ofta är abstrakt formulerad.
    """
    T = antal_teman
    U = prototyp.shape[0] - T
    ut = np.zeros((U, prototyp.shape[1]), dtype=bool)
    tema_max = prototyp[:T].max(axis=0) if T else np.zeros(prototyp.shape[1])
    for j in range(U):
        niva = exempelniva[T + j]
        if niva is None:
            continue
        e = prototyp[T + j]
        ut[j] = (e >= UNDANTAGSNIVA * niva) & (e >= UNDANTAGSMARGINAL * tema_max)
    return ut


def kalibrera(S: np.ndarray, P: list[np.ndarray], minst: int = 5) -> tuple[np.ndarray, list]:
    """Gör om rå likhet till relevans 0–1 för varje fråga.

    0 motsvarar en typisk publikation bland kandidaterna (medianen, som för ett
    enskilt tema nästan alltid är orelaterad). 1 motsvarar en fullträff:
    medianen bland de bästa sökträffarna för just den frågan. För allmänt hållna
    teman sätter den medianen ribban för lågt. Då höjs ribban till den hundradel
    av kandidaternas verk som liknar frågan mest, men aldrig över vad den bästa
    tiondelen av sökträffarna når.
    Skalan blir jämförbar mellan teman och mellan språkmodell och ordjämförelse.
    """
    G = np.zeros_like(S, dtype=np.float64)
    ankare = []
    for q in range(S.shape[0]):
        rad = S[q]
        if not rad.size:
            ankare.append(None)
            continue
        botten = float(np.median(rad))
        pos = np.asarray(P[q])[:25] if q < len(P) else np.zeros(0)
        topp = float(np.percentile(rad, 99))
        if len(pos) >= minst:
            topp = max(float(np.median(pos)), min(topp, float(np.percentile(pos, 90))))
        if topp - botten < 1e-6:
            ankare.append(None)
            continue
        G[q] = np.clip((rad - botten) / (topp - botten), 0, 1)
        ankare.append((botten, topp))
    return G, ankare


# ---------------------------------------------------------------- bedömning
def bedom(verk: list[dict], G_tema: np.ndarray, undantag: np.ndarray,
          exkl_lander: frozenset | set = frozenset()) -> dict:
    """Sammanfattar en forskares verk. G_tema har en rad per tema och en kolumn
    per verk (relevans 0–1). `undantag` har en rad per undantag och säger vilka
    verk som hör till det undantagna området (se undantagsverk)."""
    T = G_tema.shape[0]
    djup = np.zeros(T)
    antal = np.zeros(T, dtype=int)
    relevanta = antal_undantag = sista5 = bas_exkl = med_exkl = 0
    medf: Counter = Counter()
    topp: list[tuple] = []
    for j, v in enumerate(verk):
        g = G_tema[:, j]
        vikt = rollvikt(v["pos"], v["n"], v["kor"]) * titelvikt(v["t"]) * aktualitet(v["y"])
        # bidrag: 0 vid GOLV, 0,5 vid RELEVANT, 1 från 0,8 och uppåt
        c = np.clip((g - GOLV) / (2 * (RELEVANT - GOLV)), 0, 1)
        djup += c * vikt
        antal += g >= RELEVANT
        if T and float(g.max()) >= RELEVANT:
            relevanta += 1
        if undantag.size and undantag[:, j].any():
            antal_undantag += 1
        if T and float(c.max()) > 0:
            topp.append((float(c.max()) * vikt, v["t"], v["y"], v["doi"]))
        nytt = (v["y"] or 0) >= AR - AKTUELLT_AR
        if nytt and (v["n"] <= 1 or v["pos"] == "last"):
            sista5 += 1
        if nytt and 2 <= v["n"] <= MANGA_FORFATTARE:
            bas_exkl += 1
            if exkl_lander and exkl_lander & set(v["ml"]):
                med_exkl += 1
        for m in v["med"]:
            medf[m] += 1
    topp.sort(key=lambda x: -x[0])
    return {
        "styrka": (1 - np.exp(-djup / MATTNAD)).tolist(),
        "antal": antal.tolist(),
        "relevanta": relevanta,
        "undantag_andel": antal_undantag / len(verk) if verk else 0.0,
        "sista5": sista5,
        "andel_exkl": (med_exkl / bas_exkl) if (exkl_lander and bas_exkl) else None,
        "medf": medf,
        "toppverk": topp[:3],
    }
