"""Liten OpenAlex-klient med diskcache, omförsök och hastighetsbegränsning.

API-nyckel (gratis): skapa konto på openalex.org och kopiera nyckeln från
https://openalex.org/settings/api. Lägg den i miljövariabeln OPENALEX_API_KEY
eller i filen .env bredvid skripten (OPENALEX_API_KEY=...).

Utan nyckel fungerar anropen, men dagsbudgeten är 0,10 dollar och delas med
alla på samma nätverk. Med nyckel är den 1 dollar per dag. En sökning kostar
0,001 dollar och ett listanrop 0,0001 dollar.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests

BASE = "https://api.openalex.org"
HERE = Path(__file__).resolve().parent


class OpenAlexStopp(SystemExit):
    """Fel som inte går att försöka om: ogiltig nyckel eller slut på dagsbudgeten.

    Ärver SystemExit så att kommandoradsprogrammen avslutas med ett rent
    meddelande. Appen fångar felet och visar det i stället.
    """

# Länder i Europa (ISO 3166-1 alpha-2), inklusive länder utanför EU.
EUROPA = {
    "AD", "AL", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK", "EE",
    "ES", "FI", "FO", "FR", "GB", "GG", "GI", "GR", "HR", "HU", "IE", "IM", "IS",
    "IT", "JE", "LI", "LT", "LU", "LV", "MC", "MD", "ME", "MK", "MT", "NL", "NO",
    "PL", "PT", "RO", "RS", "SE", "SI", "SK", "SM", "UA", "VA", "XK",
}


def _las_api_nyckel() -> str | None:
    nyckel = os.environ.get("OPENALEX_API_KEY")
    if nyckel:
        return nyckel.strip()
    env = HERE / ".env"
    if env.exists():
        for rad in env.read_text(encoding="utf-8").splitlines():
            if rad.strip().startswith("OPENALEX_API_KEY="):
                return rad.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class OpenAlex:
    def __init__(self, cache_dir: Path | None = None, api_key: str | None = None,
                 session: requests.Session | None = None):
        self.api_key = api_key or _las_api_nyckel()
        if not self.api_key:
            print("VARNING: ingen OPENALEX_API_KEY hittades. Utan nyckel är dagsbudgeten "
                  "0,10 dollar, vilket räcker till ungefär en snabb körning.", file=sys.stderr)
        self.cache_dir = cache_dir or (HERE / ".cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self._las = threading.Lock()
        self._senaste_anrop = 0.0
        self._senaste_semantiska = 0.0
        self.antal_anrop = 0
        self.kostnad_usd = 0.0          # vad den här körningen har kostat
        self.kvar_usd: float | None = None    # kvar av dagsbudgeten enligt senaste svar
        self.grans_usd: float | None = None   # dagsbudgetens storlek

    # -- intern hjälp -----------------------------------------------------
    def _cachefil(self, path: str, params: dict, tagg: str = "") -> Path:
        nyckel = json.dumps([path, sorted(params.items())] + ([tagg] if tagg else []),
                            ensure_ascii=False)
        # Svaren sparas komprimerade: ett söksvar med abstracts är annars runt 1 MB.
        return self.cache_dir / (hashlib.sha1(nyckel.encode()).hexdigest() + ".json.gz")

    @staticmethod
    def _las_cache(fil: Path) -> dict:
        if fil.suffix == ".gz":
            return json.loads(gzip.decompress(fil.read_bytes()).decode("utf-8"))
        return json.loads(fil.read_text(encoding="utf-8"))

    @staticmethod
    def _skriv_cache(fil: Path, data: dict):
        text = json.dumps(data, ensure_ascii=False)
        # Skriv först till en egen fil och byt sedan namn, så att ett avbrott
        # (eller en parallell tråd) aldrig lämnar en halv cachefil efter sig.
        tmp = fil.with_name(f"{fil.name}.{threading.get_ident()}.tmp")
        tmp.write_bytes(gzip.compress(text.encode("utf-8")))
        tmp.replace(fil)

    def _bokfor(self, svar: requests.Response):
        def tal(namn):
            try:
                return float(svar.headers.get(namn))
            except (TypeError, ValueError):
                return None
        with self._las:
            self.antal_anrop += 1
            self.kostnad_usd += tal("X-RateLimit-Cost-USD") or 0.0
            kvar, grans = tal("X-RateLimit-Remaining-USD"), tal("X-RateLimit-Limit-USD")
            if kvar is not None:
                self.kvar_usd = kvar
            if grans is not None:
                self.grans_usd = grans

    def _budget_slut(self, svar: requests.Response, pris: float) -> bool:
        """Avgör om ett 429-svar betyder att dagsbudgeten är slut (då hjälper det
        inte att vänta några sekunder) eller bara att vi anropar för tätt.
        `pris` är vad anropet kostar: är mindre än så kvar är budgeten slut."""
        try:
            return float(svar.headers.get("X-RateLimit-Remaining-USD")) < pris
        except (TypeError, ValueError):
            pass    # ingen uppgift om budgeten i svaret – gissa utifrån resten
        try:
            if float(svar.headers.get("Retry-After")) > 120:
                return True
        except (TypeError, ValueError):
            pass
        text = svar.text[:600].lower()
        return any(o in text for o in ("budget", "daily", "per day"))

    def get(self, path: str, params: dict | None = None, use_cache: bool = True,
            bearbeta=None, tagg: str = "") -> dict:
        """Hämtar ett svar, från cachen om det finns där.

        `bearbeta` är en funktion som krymper svaret innan det sparas. Då måste
        `tagg` anges, så att det krympta svaret får en egen plats i cachen.
        """
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cachefil = self._cachefil(path, params, tagg)
        if use_cache:
            # .json.gz, eller okomprimerad .json från äldre versioner av verktyget
            for fil in (cachefil, cachefil.with_suffix("")):
                if fil.exists():
                    try:
                        return self._las_cache(fil)
                    except (OSError, ValueError):
                        pass    # trasig cachefil – hämta om

        # Semantisk sökning tillåter 1 anrop/s, övriga max ~10/s härifrån.
        # Tiden bokas under lås så att parallella trådar inte anropar samtidigt.
        semantisk = "search.semantic" in params
        with self._las:
            start = max(time.time(), self._senaste_anrop + 0.1)
            if semantisk:
                start = max(start, self._senaste_semantiska + 1.05)
                self._senaste_semantiska = start
            self._senaste_anrop = start
        paus = start - time.time()
        if paus > 0:
            time.sleep(paus)

        fullparams = dict(params)
        if self.api_key:
            fullparams["api_key"] = self.api_key

        for forsok in range(6):
            try:
                svar = self.session.get(f"{BASE}{path}", params=fullparams, timeout=90)
            except requests.RequestException:
                if forsok == 5:
                    raise
                time.sleep(2 ** forsok)
                continue
            self._bokfor(svar)
            if svar.status_code == 200:
                data = svar.json()
                if bearbeta:
                    data = bearbeta(data)
                self._skriv_cache(cachefil, data)
                return data
            pris = 0.001 if ("search" in params or semantisk) else 0.0001
            if svar.status_code == 429 and self._budget_slut(svar, pris):
                raise OpenAlexStopp(
                    "Dagens OpenAlex-budget är slut. Den fylls på igen nästa dygn. "
                    + ("" if self.api_key else
                       "Du kör utan API-nyckel (0,10 dollar per dag, delat med andra på nätverket). "
                         "Med en gratis nyckel från https://openalex.org/settings/api får du 1 dollar per dag. ")
                    + "Det som redan hämtats ligger kvar i cachen, så nästa körning fortsätter där den här slutade.")
            if svar.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** forsok + 1)
                continue
            if svar.status_code in (401, 403, 409):
                raise OpenAlexStopp(
                    f"OpenAlex svarade {svar.status_code}: {svar.text[:300]}\n"
                    "Kontrollera API-nyckeln och dagskvoten "
                    "(https://api.openalex.org/rate-limit?api_key=DIN_NYCKEL).")
            raise RuntimeError(f"OpenAlex {svar.status_code} för {path}: {svar.text[:300]}")
        raise RuntimeError(f"OpenAlex svarade inte efter flera försök: {path}")

    def kolla_budget(self) -> float | None:
        """Läser av hur mycket av dagsbudgeten som återstår. Anropet är gratis."""
        try:
            self.get("/topics/T10762", use_cache=False)
        except (RuntimeError, requests.RequestException):
            return None
        return self.kvar_usd

    # -- bekväma anrop -----------------------------------------------------
    def semantisk_sokning(self, text: str, filter_: str, select: str, antal: int = 50) -> list[dict]:
        data = self.get("/works", {
            "search.semantic": text[:1500],
            "filter": filter_,
            "select": select,
            "per_page": min(antal, 50),
        })
        return data.get("results", [])

    def nyckelordssokning(self, text: str, filter_: str, select: str,
                          antal: int = 200) -> list[dict]:
        resultat: list[dict] = []
        sida = 1
        while len(resultat) < antal:
            data = self.get("/works", {
                "search": text,
                "filter": filter_,
                "select": select,
                "per_page": min(100, antal),
                "page": sida,
            })
            traffar = data.get("results", [])
            resultat.extend(traffar)
            if len(traffar) < min(100, antal):
                break
            sida += 1
        return resultat[:antal]

    def hamta_forfattare(self, ids: list[str], select: str) -> list[dict]:
        """Hämtar författarprofiler i grupper om 50 (billiga listanrop)."""
        ut: list[dict] = []
        korta = [kort_id(i) for i in ids]
        for start in range(0, len(korta), 50):
            grupp = korta[start:start + 50]
            data = self.get("/authors", {
                "filter": "openalex:" + "|".join(grupp),
                "select": select,
                "per_page": 50,
            })
            ut.extend(data.get("results", []))
        return ut

    def forsta_publiceringsar(self, author_id: str) -> int | None:
        data = self.get("/works", {
            "filter": f"author.id:{kort_id(author_id)},publication_year:>1950",
            "sort": "publication_year",
            "select": "publication_year",
            "per_page": 1,
        })
        res = data.get("results", [])
        return res[0].get("publication_year") if res else None

    def verk_for_forfattare(self, author_id: str, fran_ar: int, select: str,
                            antal: int = 100) -> list[dict]:
        data = self.get("/works", {
            "filter": f"author.id:{kort_id(author_id)},publication_year:>{fran_ar - 1}",
            "sort": "publication_year:desc",
            "select": select,
            "per_page": min(antal, 100),
        })
        return data.get("results", [])

    def verk_kompakt(self, author_id: str, fran_ar: int, antal: int = 100) -> dict:
        """En forskares senaste verk i krympt form, för profilkontrollen i rekrytera.py.

        Ett rått svar är 1–2 MB per forskare. Här sparas bara det som används:
        titel, abstract, år, forskarens egen position och lärosäte samt vilka
        medförfattarna är. Ger {"totalt": antal verk i perioden, "verk": [...]}.
        """
        aid = kort_id(author_id)
        return self.get("/works", {
            "filter": (f"author.id:{aid},publication_year:>{fran_ar - 1},"
                       "type:article|review|book|book-chapter"),
            "sort": "publication_year:desc",
            "select": "id,doi,title,publication_year,type,authorships,abstract_inverted_index",
            "per_page": min(antal, 100),
        }, bearbeta=lambda data: _krymp_verk(data, aid), tagg="kompakt1")

    def forfattare_via_orcid(self, orcid: str) -> dict | None:
        orcid = orcid.strip().replace("https://orcid.org/", "")
        try:
            return self.get(f"/authors/orcid:{orcid}", {
                "select": "id,display_name,last_known_institutions"})
        except RuntimeError:
            return None


def kort_id(openalex_id: str) -> str:
    return (openalex_id or "").rsplit("/", 1)[-1]


MANGA_FORFATTARE = 30   # över detta sparas inte medförfattarna (konsortieartiklar)


def _krymp_verk(data: dict, aid: str) -> dict:
    """Krymper ett /works-svar för forskaren `aid` (kort id). Se verk_kompakt."""
    ut = []
    for v in data.get("results") or []:
        aus = v.get("authorships") or []
        pos, kor, inst, med, lander = None, False, [], [], set()
        for a in aus:
            a_id = kort_id((a.get("author") or {}).get("id") or "")
            if a_id == aid:
                pos = a.get("author_position")
                kor = bool(a.get("is_corresponding"))
                inst = [[kort_id(i.get("id") or ""), i.get("display_name") or "",
                         (i.get("country_code") or "").upper()]
                        for i in a.get("institutions") or []]
                continue
            if a_id:
                med.append(a_id)
            lander.update(l.upper() for l in a.get("countries") or [])
            lander.update((i.get("country_code") or "").upper() for i in a.get("institutions") or [])
        fa = len(aus) <= MANGA_FORFATTARE
        ut.append({
            "id": kort_id(v.get("id") or ""), "doi": v.get("doi"), "t": v.get("title") or "",
            "y": v.get("publication_year"), "ty": v.get("type"),
            "a": abstrakt_text(v.get("abstract_inverted_index"))[:2500],
            "n": len(aus), "pos": pos, "kor": kor, "inst": inst,
            "med": med if fa else [], "ml": sorted(lander - {""}) if fa else [],
        })
    return {"totalt": (data.get("meta") or {}).get("count"), "verk": ut}


def abstrakt_text(inverterat: dict | None) -> str:
    """OpenAlex lagrar abstracts som inverterat index; bygg tillbaka texten."""
    if not inverterat:
        return ""
    positioner: list[tuple[int, str]] = []
    for ord_, pos in inverterat.items():
        for p in pos:
            positioner.append((p, ord_))
    positioner.sort()
    return " ".join(o for _, o in positioner)
