"""Liten OpenAlex-klient med diskcache, omförsök och hastighetsbegränsning.

API-nyckel (gratis) krävs sedan februari 2026: skapa konto på openalex.org och
kopiera nyckeln från https://openalex.org/settings/api. Lägg den i miljövariabeln
OPENALEX_API_KEY eller i filen .env bredvid skripten (OPENALEX_API_KEY=...).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = "https://api.openalex.org"
HERE = Path(__file__).resolve().parent

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
            print("VARNING: ingen OPENALEX_API_KEY hittades. Utan nyckel räcker "
                  "dagskvoten bara till några få anrop.", file=sys.stderr)
        self.cache_dir = cache_dir or (HERE / ".cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self._senaste_anrop = 0.0
        self._senaste_semantiska = 0.0
        self.antal_anrop = 0

    # -- intern hjälp -----------------------------------------------------
    def _cachefil(self, path: str, params: dict) -> Path:
        nyckel = json.dumps([path, sorted(params.items())], ensure_ascii=False)
        return self.cache_dir / (hashlib.sha1(nyckel.encode()).hexdigest() + ".json")

    def get(self, path: str, params: dict | None = None, use_cache: bool = True) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cachefil = self._cachefil(path, params)
        if use_cache and cachefil.exists():
            return json.loads(cachefil.read_text(encoding="utf-8"))

        # Semantisk sökning tillåter 1 anrop/s, övriga max ~10/s härifrån.
        semantisk = "search.semantic" in params
        vanta = 1.05 if semantisk else 0.1
        senaste = self._senaste_semantiska if semantisk else self._senaste_anrop
        paus = vanta - (time.time() - senaste)
        if paus > 0:
            time.sleep(paus)

        fullparams = dict(params)
        if self.api_key:
            fullparams["api_key"] = self.api_key

        for forsok in range(6):
            try:
                svar = self.session.get(f"{BASE}{path}", params=fullparams, timeout=60)
            except requests.RequestException as fel:
                if forsok == 5:
                    raise
                time.sleep(2 ** forsok)
                continue
            self.antal_anrop += 1
            self._senaste_anrop = time.time()
            if semantisk:
                self._senaste_semantiska = time.time()
            if svar.status_code == 200:
                data = svar.json()
                cachefil.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
            if svar.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** forsok + 1)
                continue
            if svar.status_code in (401, 403, 409):
                raise SystemExit(
                    f"OpenAlex svarade {svar.status_code}: {svar.text[:300]}\n"
                    "Kontrollera API-nyckeln och dagskvoten "
                    "(https://api.openalex.org/rate-limit?api_key=DIN_NYCKEL).")
            raise RuntimeError(f"OpenAlex {svar.status_code} för {path}: {svar.text[:300]}")
        raise RuntimeError(f"OpenAlex svarade inte efter flera försök: {path}")

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

    def forfattare_via_orcid(self, orcid: str) -> dict | None:
        orcid = orcid.strip().replace("https://orcid.org/", "")
        try:
            return self.get(f"/authors/orcid:{orcid}", {
                "select": "id,display_name,last_known_institutions"})
        except RuntimeError:
            return None


def kort_id(openalex_id: str) -> str:
    return openalex_id.rsplit("/", 1)[-1]


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
