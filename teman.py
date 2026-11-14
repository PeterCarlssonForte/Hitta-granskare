"""Teman: de kompetenser beredningsgruppen behöver, i ett enkelt textformat.

En rad per tema:

    kort sökfras | beskrivning

Sökfrasen (2–6 ord) används i nyckelordssökningen och som etikett.
Beskrivningen (2–3 meningar, skriven som ett abstract) används i den semantiska
sökningen och när varje kandidats publikationer jämförs med temat.

Rader som börjar med "EJ:" är undantag: forskning som utlysningen inte
finansierar. De används aldrig för att hitta kandidater, bara för att sänka
poängen för den som mest forskar om sådant.

Rader som börjar med # är kommentarer. Det äldre formatet, med bara en fras
per rad, fungerar fortfarande.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

LANG_FRAS = 120     # längre rader utan "|" tolkas som en beskrivning utan sökfras

_UNDANTAG = re.compile(r"^(?:ej|undantag|exkludera)\s*:\s*", re.IGNORECASE)
_PUNKT = re.compile(r"^\s*(?:[-•*–]+|\d+[.)])\s+")
_BARA_STRECK = re.compile(r"[-|:\s]+")


@dataclass(frozen=True)
class Tema:
    fras: str = ""
    beskrivning: str = ""

    @property
    def etikett(self) -> str:
        if self.fras:
            return self.fras
        ord_ = self.beskrivning.split()
        return " ".join(ord_[:8]) + ("…" if len(ord_) > 8 else "")

    @property
    def soktext(self) -> str:
        """Texten för semantisk sökning: beskrivningen om den finns."""
        return self.beskrivning or self.fras

    @property
    def fulltext(self) -> str:
        """Texten som kandidaternas publikationer jämförs med."""
        return ". ".join(x.rstrip(".") for x in (self.fras, self.beskrivning) if x)


def tolka_teman(text: str) -> tuple[list[Tema], list[Tema]]:
    """Tolkar temaformatet. Ger (teman, undantag)."""
    teman: list[Tema] = []
    undantag: list[Tema] = []
    for rad in (text or "").splitlines():
        rad = rad.strip()
        if not rad or rad.startswith("#") or rad.startswith("```"):
            continue
        # Tål punktlistor, numrering, fetstil och tabellstreck från en AI-assistent.
        rad = _PUNKT.sub("", rad).replace("**", "").strip().strip("|").strip()
        if not rad or _BARA_STRECK.fullmatch(rad):
            continue
        m = _UNDANTAG.match(rad)
        if m:
            rad = rad[m.end():].strip()
        fras, _, beskrivning = rad.partition("|")
        fras, beskrivning = fras.strip(), beskrivning.strip()
        if not beskrivning and len(fras) > LANG_FRAS:
            fras, beskrivning = "", fras
        if fras or beskrivning:
            (undantag if m else teman).append(Tema(fras, beskrivning))
    return teman, undantag


def las_temafil(fil: Path) -> tuple[list[Tema], list[Tema]]:
    return tolka_teman(Path(fil).read_text(encoding="utf-8-sig"))


def teman_till_text(teman: list[Tema], undantag: list[Tema] = ()) -> str:
    def rad(t: Tema) -> str:
        return f"{t.fras} | {t.beskrivning}" if t.fras and t.beskrivning else (t.fras or t.beskrivning)
    return "\n".join([rad(t) for t in teman] + [f"EJ: {rad(u)}" for u in undantag])


def ser_svensk_ut(text: str) -> bool:
    ord_ = re.findall(r"\b\w+\b", text.lower())
    svenska = {"och", "att", "som", "för", "med", "det", "inom", "av", "på", "är"}
    return sum(o in svenska for o in ord_) >= max(2, len(ord_) * 0.05)


def granska_teman(teman: list[Tema], undantag: list[Tema] = ()) -> list[str]:
    """Påpekanden om temalistan, på svenska. Tom lista betyder att allt ser bra ut."""
    if not teman:
        return ["Skriv minst ett tema. Sökningen utgår helt från temana."]
    varningar = []
    if len(teman) < 4:
        varningar.append(f"Bara {len(teman)} teman. Med 5–8 teman täcker gruppen fler kompetenser.")
    if len(teman) > 12:
        varningar.append(f"{len(teman)} teman ger många sökningar och en dyr körning. Slå gärna ihop närliggande teman.")
    utan_beskrivning = [t.etikett for t in teman if not t.beskrivning]
    if utan_beskrivning:
        varningar.append("Saknar beskrivning: " + "; ".join(utan_beskrivning[:4])
                         + (" med flera" if len(utan_beskrivning) > 4 else "")
                         + ". Den semantiska sökningen träffar bättre med två–tre beskrivande meningar efter ett |.")
    utan_fras = [t.etikett for t in teman if not t.fras]
    if utan_fras:
        varningar.append("Saknar kort sökfras: " + "; ".join(utan_fras[:4])
                         + ". Utan fras görs ingen nyckelordssökning för temat.")
    if any(ser_svensk_ut(t.fulltext) for t in list(teman) + list(undantag)):
        varningar.append("Några teman ser ut att vara på svenska. Publikationerna i OpenAlex är nästan "
                         "bara på engelska, så skriv temana på engelska.")
    return varningar


def ai_prompt(utlysning: str) -> str:
    """En färdig fråga att klistra in i en AI-assistent. Svaret är en temalista
    i formatet ovan, som sedan redigeras för hand."""
    return f"""Du hjälper en forskningsfinansiär att rekrytera sakkunniga granskare till utlysningen nedan.
Ta fram de kompetenser som beredningsgruppen sammantaget behöver för att kunna bedöma ansökningarna.

Skriv 5–8 teman. Varje tema är en kompetens: ett ämnesområde, en metod eller en målgrupp/arena.
Skriv allt på engelska, eftersom temana används för att söka i en internationell publikationsdatabas.

Format, en rad per tema:
kort sökfras | beskrivning

- Sökfrasen: 2–6 ord, så som området brukar benämnas i artikeltitlar.
- Beskrivningen: 2–3 meningar som beskriver forskningen inom temat så som den beskrivs i ett abstract:
  vad som studeras, i vilka grupper eller miljöer och med vilka metoder. Beskriv forskningen, inte utlysningen.
- Ta inte med krav som gäller själva ansökan (budget, samverkansplan, formalia).

Lägg sist till en rad för varje forskningsområde som utlysningen uttryckligen INTE finansierar, i formatet:
EJ: kort fras | beskrivning

Svara bara med raderna, utan rubriker, numrering eller kommentarer.

UTLYSNING:
{utlysning.strip()}
"""
