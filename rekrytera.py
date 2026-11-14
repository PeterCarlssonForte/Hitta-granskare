"""Steg 1: hitta europeiska forskare som passar en utlysning.

Exempel:
    python rekrytera.py --teman exempel/teman.txt --exkludera-lander SE --ut kandidater.xlsx

Få en färdig fråga till en AI-assistent som skriver ett temaförslag:
    python rekrytera.py --utlysning exempel/utlysning.txt --temaprompt

Flöde
 1. Teman. De kompetenser beredningsgruppen behöver skrivs som teman, ett per
    rad: "kort sökfras | beskrivning". Formatet beskrivs i teman.py.
 2. Sökning. Varje tema söks i OpenAlex: semantiskt på beskrivningen och med
    nyckelord på sökfrasen. Sökningen tar bara fram namn.
 3. Profilkontroll. För de bästa namnen hämtas deras egna publikationer. Varje
    forskare bedöms efter hur många av dem som matchar varje tema, hur nya de
    är och vilken roll forskaren hade (se profil.py).
 4. Senioritet. En grov bedömning görs (docentnivå trolig/möjlig/osäker).
    OBS: docentur finns inte som data – bedömningen måste verifieras.
 5. Gruppförslag. En beredningsgrupp föreslås där varje tema täcks av två
    personer som inte har publicerat tillsammans, med spridning över länder
    och lärosäten och med reserver för varje plats.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import profil
from openalex import EUROPA, OpenAlex, abstrakt_text, kort_id
from teman import Tema, ai_prompt, granska_teman, las_temafil

AR = dt.date.today().year

# Trösklar för senioritetsbedömningen. Justera efter ämnesområde – t.ex. har
# samhällsvetare ofta lägre h-index än medicinare på samma karriärnivå.
# "sista_forfattare" = verk de senaste fem åren som sista- eller ensamförfattare,
# ett tecken på att personen leder egen forskning.
TROLIG = {"akademisk_alder": 12, "h_index": 12, "antal_verk": 30, "sista_forfattare": 3}
MOJLIG = {"akademisk_alder": 8, "h_index": 8, "antal_verk": 15}

MIN_RELEVANTA = 3       # relevanta verk som krävs för att kunna föreslås till gruppen
MAX_UNDANTAG = 0.5      # den som har minst så stor andel verk i undantagna områden föreslås inte
TACKT = 0.5             # temastyrka från vilken en person räknas täcka temat
UNDANTAGSAVDRAG = 0.6   # så mycket sänks matchningen om alla verk ligger i undantagna områden

# Sökdjup. "perioder" delar den semantiska sökningen i tidsperioder (50 träffar
# per period), "nyckelord" är antal träffar per nyckelordssökning och "profiler"
# hur många forskare som profilkontrolleras.
DJUP = {
    "snabb": {"perioder": 1, "nyckelord": 100, "profiler": 150},
    "normal": {"perioder": 3, "nyckelord": 200, "profiler": 300},
    "grundlig": {"perioder": 3, "nyckelord": 200, "profiler": 500, "fras_semantiskt": True},
}
PRIS_SOKNING, PRIS_LISTA = 0.001, 0.0001    # dollar per anrop enligt OpenAlex

TYPFILTER = "type:article|review|book|book-chapter"
VERK_SELECT = "id,doi,title,publication_year,type,authorships,relevance_score,abstract_inverted_index"
FORF_SELECT = ("id,display_name,orcid,works_count,cited_by_count,summary_stats,"
               "last_known_institutions,topics")


class IngaTraffar(RuntimeError):
    """Sökningen gav inga användbara kandidater."""


def _tyst(text: str, andel: float | None = None):
    pass


def kort_etikett(text: str, n: int = 60) -> str:
    rad = (text.splitlines() or [""])[0]
    return rad if len(rad) <= n else rad[: n - 1] + "…"


def uppskatta_kostnad(antal_teman: int, antal_undantag: int, djup: str, antal_kandidater: int) -> float:
    """Högsta kostnad i dollar för en körning. Svar som finns i cachen är gratis."""
    d = DJUP[djup]
    semantiska = antal_teman * d["perioder"] * (2 if d.get("fras_semantiskt") else 1) + antal_undantag
    nyckelord = antal_teman * math.ceil(d["nyckelord"] / 100)
    listor = d["profiler"] + antal_kandidater + math.ceil(d["profiler"] / 50)
    return (semantiska + nyckelord) * PRIS_SOKNING + listor * PRIS_LISTA


# ---------------------------------------------------------------- sökning
def europeiska_lander(authorship: dict) -> set[str]:
    lander = set(authorship.get("countries") or [])
    for inst in authorship.get("institutions") or []:
        if inst.get("country_code"):
            lander.add(inst["country_code"])
    return {l.upper() for l in lander} & EUROPA


def perioder(fran_ar: int, antal: int) -> list[list[int] | None]:
    """Delar åren från fran_ar till i år i `antal` block, nyast först. Den
    semantiska sökningen ger högst 50 träffar per fråga, så flera perioder ger
    fler träffar för samma tema."""
    ar = list(range(fran_ar, AR + 1))
    if antal <= 1 or len(ar) < 2 * antal:
        return [None]
    storlek = math.ceil(len(ar) / antal)
    block = [ar[max(0, len(ar) - (i + 1) * storlek): len(ar) - i * storlek] for i in range(antal)]
    return [b for b in block if b]


def _verktext(verk: dict) -> str:
    text = f"{verk.get('title') or ''}. {abstrakt_text(verk.get('abstract_inverted_index'))}"
    return text.strip()[:profil.TEXTLANGD]


def samla_kandidater(oa: OpenAlex, teman: list[Tema], fran_ar: int, djup: dict,
                     framsteg=_tyst) -> tuple[dict, dict, dict]:
    """Söker varje tema och samlar europeiska författare på träffarna.

    Ger (s1, namn, positiva):
      s1[forskare][tema] – sökpoäng, summan över forskarens träffade verk. Varje
        verk räknas en gång per tema, hur många frågor det än träffas av.
      namn[forskare] – visningsnamn.
      positiva[tema] – texterna för temats 25 bästa semantiska träffar. De visar
        profilkontrollen hur en fullträff på temat ser ut.
    """
    nyckelordsfilter = (f"authorships.institutions.continent:europe,"
                        f"publication_year:>{fran_ar - 1},{TYPFILTER}")
    # (tema, slag, text, filter, antal, används för kalibrering)
    fragor: list[tuple[int, str, str, str, int, bool]] = []
    for i, tema in enumerate(teman):
        for ar in perioder(fran_ar, djup["perioder"]):
            # Semantisk sökning kan inte filtreras på land – Europa sållas fram efteråt.
            arfilter = ("publication_year:" + "|".join(map(str, ar))) if ar \
                else f"publication_year:>{fran_ar - 1}"
            fragor.append((i, "semantisk", tema.soktext, f"{arfilter},{TYPFILTER}", 50, True))
            if djup.get("fras_semantiskt") and tema.fras and tema.beskrivning:
                fragor.append((i, "semantisk", tema.fras, f"{arfilter},{TYPFILTER}", 50, False))
        if tema.fras:
            fragor.append((i, "nyckelord", tema.fras, nyckelordsfilter, djup["nyckelord"], False))

    basta: dict = defaultdict(lambda: defaultdict(dict))    # forskare → tema → verk → vikt
    namn: dict[str, str] = {}
    positiva: dict[int, list] = defaultdict(list)
    for nr, (i, slag, text, filter_, antal, kalibrering) in enumerate(fragor, 1):
        framsteg(f"Söker [{nr}/{len(fragor)}] {slag} – {kort_etikett(teman[i].etikett, 50)}",
                 nr / len(fragor))
        try:
            if slag == "semantisk":
                traffar = oa.semantisk_sokning(text, filter_, VERK_SELECT, antal)
            else:
                traffar = oa.nyckelordssokning(text, filter_, VERK_SELECT, antal)
        except RuntimeError as fel:
            framsteg(f"    hoppar över frågan: {fel}")
            continue
        halvering = 25 if slag == "semantisk" else 50
        for rang, verk in enumerate(traffar):
            titel = verk.get("title") or ""
            typvikt = profil.titelvikt(titel)
            if typvikt == 0:
                continue
            if kalibrering:
                positiva[i].append((verk.get("relevance_score") or 0.0, _verktext(verk)))
            nyckel = profil.norm_titel(titel) or verk.get("id")
            forfattare = verk.get("authorships") or []
            for au in forfattare:
                aid = kort_id((au.get("author") or {}).get("id") or "")
                if not aid or not europeiska_lander(au):
                    continue
                vikt = (typvikt / (1 + rang / halvering)
                        * profil.rollvikt(au.get("author_position"), len(forfattare),
                                          bool(au.get("is_corresponding"))))
                namn[aid] = (au.get("author") or {}).get("display_name") or namn.get(aid, "")
                if vikt > basta[aid][i].get(nyckel, 0.0):
                    basta[aid][i][nyckel] = vikt

    s1 = {aid: {i: sum(verk.values()) for i, verk in per_tema.items()}
          for aid, per_tema in basta.items()}
    pos = {i: [text for _, text in sorted(positiva[i], key=lambda x: -x[0])[:25]]
           for i in range(len(teman))}
    return s1, namn, pos


def valj_for_profil(s1: dict, antal: int, antal_teman: int) -> list[str]:
    """Väljer vilka forskare som profilkontrolleras: de bästa per tema, så att
    inget tema blir utan kandidater, och sedan de bästa totalt."""
    total = {aid: sum(per_tema.values()) for aid, per_tema in s1.items()}
    per_tema = max(10, antal // (2 * max(1, antal_teman)))
    valda: set[str] = set()
    for t in range(antal_teman):
        i_temat = sorted((a for a in s1 if t in s1[a]), key=lambda a: (-s1[a][t], a))
        valda.update(i_temat[:per_tema])
    for aid in sorted(total, key=lambda a: (-total[a], a)):
        if len(valda) >= antal:
            break
        valda.add(aid)
    return sorted(valda, key=lambda a: (-total[a], a))[:antal]


# ---------------------------------------------------------------- person
def hemvist(prof: dict, verk: list[dict], exkl: set[str] = frozenset()) -> tuple[dict, set[str]]:
    """Forskarens lärosäte och de länder hen är verksam i.

    OpenAlex anger ofta flera "senast kända" lärosäten. Här väljs det som
    förekommer oftast i forskarens tio senaste publikationer. Väger det jämnt
    väljs ett europeiskt lärosäte i ett land som inte är uteslutet, så att
    personen syns i listan (med varning) i stället för att försvinna.
    Länderna är alla senast kända plus de som anges i publikationer från i år
    och i fjol.
    """
    senast = [i for i in prof.get("last_known_institutions") or [] if i]
    forekomst: Counter = Counter()
    namn_land: dict[str, tuple[str, str]] = {}
    nya_lander: set[str] = set()
    raknade = 0
    for v in verk:      # nyast först
        if not v["inst"]:
            continue
        for iid, inamn, land in v["inst"]:
            forekomst[iid] += 1
            namn_land.setdefault(iid, (inamn, land))
            if (v["y"] or 0) >= AR - 1:
                nya_lander.add(land)
        raknade += 1
        if raknade >= 10:
            break
    if senast:
        def lamplighet(i: dict) -> tuple:
            land = (i.get("country_code") or "").upper()
            return (forekomst.get(kort_id(i.get("id") or ""), 0), land in EUROPA and land not in exkl)
        bast = max(senast, key=lamplighet)
        inst = {"namn": bast.get("display_name") or "", "land": (bast.get("country_code") or "").upper()}
    elif forekomst:
        inamn, land = namn_land[forekomst.most_common(1)[0][0]]
        inst = {"namn": inamn, "land": land}
    else:
        inst = {"namn": "", "land": ""}
    lander = {(i.get("country_code") or "").upper() for i in senast} | nya_lander
    return inst, lander - {""}


def matchning(styrka: list[float], undantag_andel: float | None) -> float:
    """0–100. Djup i det starkaste temat ger två tredjedelar av poängen, bredd
    resten: starkaste temat + en tredjedel av näst starkaste + en sjättedel av
    det tredje. Sänks om en stor del av forskarens verk ligger i undantagna
    områden."""
    s = sorted(styrka, reverse=True) + [0.0, 0.0, 0.0]
    bas = 100 * (s[0] + s[1] / 3 + s[2] / 6) / 1.5
    return bas * (1 - UNDANTAGSAVDRAG * (undantag_andel or 0.0))


def senioritet(akad_alder: int | None, h: int | None, verk: int | None, sista: int | None = None) -> str:
    """Trolig kräver, utöver trösklarna, tecken på självständighet: verk som
    sista- eller ensamförfattare de senaste fem åren. Utan `sista` (None)
    bedöms bara trösklarna."""
    a, h, w = akad_alder or 0, h or 0, verk or 0
    trolig = a >= TROLIG["akademisk_alder"] and h >= TROLIG["h_index"] and w >= TROLIG["antal_verk"]
    mojlig = a >= MOJLIG["akademisk_alder"] and h >= MOJLIG["h_index"] and w >= MOJLIG["antal_verk"]
    if trolig and (sista is None or sista >= TROLIG["sista_forfattare"]):
        return "Trolig"
    if trolig or mojlig:
        return "Möjlig"
    return "Osäker/yngre"


def profilvarning(akad_alder: int | None, verk: int | None) -> list[str]:
    varn = []
    if akad_alder and akad_alder > 50:
        varn.append("mycket lång publiceringshistorik – profilen kan vara sammanslagen")
    if verk and verk > 1500:
        varn.append("ovanligt många verk – kontrollera profilen")
    return varn


# ---------------------------------------------------------------- gruppförslag
def _sampublicerat(a: dict, b: dict) -> int:
    """Antal gemensamma verk (konsortieartiklar med över 30 författare räknas inte)."""
    return max(a["medf"].get(b["id"], 0), b["medf"].get(a["id"], 0))


def _tackningsvinst(par: list[float], styrka: float) -> float:
    """Hur mycket temats täckning ökar om en person med `styrka` läggs till.
    Täckningen är medelvärdet av de två starkaste i gruppen, så att varje
    tema helst ska bäras av två personer."""
    nya = sorted(par + [styrka], reverse=True)[:2]
    return (sum(nya) - sum(par)) / 2


def foresla_grupp(kandidater: list[dict], antal_teman: int, storlek: int,
                  max_per_land: int) -> list[dict]:
    valbara = [k for k in kandidater if k["valbar"]]
    valda: list[dict] = []
    valda_id: set[str] = set()
    tackning = [[0.0, 0.0] for _ in range(antal_teman)]
    lander: Counter = Counter()
    larosaten: Counter = Counter()
    while len(valda) < storlek:
        bast, bast_varde = None, -1e9
        for k in valbara:
            if k["id"] in valda_id or lander[k["land"]] >= max_per_land:
                continue
            varde = k["poang"] / 100
            for t in range(antal_teman):
                varde += 1.5 * _tackningsvinst(tackning[t], min(1.0, k["temastyrka"].get(t, 0.0)))
            if k["senioritet"] == "Möjlig":
                varde -= 0.15
            if larosaten[k["larosate"]]:
                varde -= 0.5
            for g in valda:     # oberoende: avdrag för den som publicerat med någon redan vald
                gemensamma = _sampublicerat(k, g)
                if gemensamma:
                    varde -= 1.0 if gemensamma >= 3 else 0.5
            varde -= 0.5 * (k["andel_exkl"] or 0.0)     # tidig jävsindikator
            if varde > bast_varde:
                bast, bast_varde = k, varde
        if bast is None:
            break
        valda.append(bast)
        valda_id.add(bast["id"])
        lander[bast["land"]] += 1
        larosaten[bast["larosate"]] += 1
        for t in range(antal_teman):
            s = min(1.0, bast["temastyrka"].get(t, 0.0))
            tackning[t] = sorted(tackning[t] + [s], reverse=True)[:2]
    return valda


def valj_reserver(grupp: list[dict], kandidater: list[dict], max_per_land: int,
                  per_plats: int = 2) -> dict[str, list[dict]]:
    """För varje plats: de valbara kandidater vars temaprofil liknar ledamotens
    mest och som inte har publicerat med någon av de övriga i gruppen."""
    tagna = {k["id"] for k in grupp}
    reserver: dict[str, list[dict]] = {}
    for ledamot in grupp:
        ovriga = [g for g in grupp if g is not ledamot]
        lander = Counter(g["land"] for g in ovriga)
        larosaten = {g["larosate"] for g in ovriga}
        summa = sum(ledamot["temastyrka"].values()) or 1.0
        forslag = []
        for k in kandidater:
            if k["id"] in tagna or not k["valbar"] or lander[k["land"]] >= max_per_land:
                continue
            if any(_sampublicerat(k, g) for g in ovriga):
                continue
            likhet = sum(min(s, k["temastyrka"].get(t, 0.0))
                         for t, s in ledamot["temastyrka"].items()) / summa
            varde = likhet + 0.3 * k["poang"] / 100 - 0.5 * (k["andel_exkl"] or 0.0)
            if k["larosate"] in larosaten:
                varde -= 0.3
            if k["senioritet"] == "Möjlig":
                varde -= 0.1
            forslag.append((varde, k))
        forslag.sort(key=lambda x: -x[0])
        reserver[ledamot["id"]] = [k for _, k in forslag[:per_plats]]
        tagna.update(k["id"] for k in reserver[ledamot["id"]])
    return reserver


def tunt_tackta(grupp: list[dict], antal_teman: int) -> list[tuple[int, int]]:
    """Teman som färre än två i gruppen täcker: [(tema, antal som täcker)]."""
    ut = []
    for t in range(antal_teman):
        n = sum(1 for k in grupp if k["temastyrka"].get(t, 0.0) >= TACKT)
        if n < 2:
            ut.append((t, n))
    return ut


# ---------------------------------------------------------------- huvudflöde
def kor(oa: OpenAlex, teman: list[Tema], undantag: list[Tema] = (), *, fran_ar: int | None = None,
        djup: str = "normal", antal_kandidater: int = 150, exkludera: list[str] = (),
        gruppstorlek: int = 10, max_per_land: int = 2, likhet: str = "auto", framsteg=None) -> dict:
    """Hela flödet från teman till kandidatlista och gruppförslag.

    `framsteg(text, andel)` anropas under körningen; andel är 0–1 eller None.
    """
    if not teman:
        raise IngaTraffar("Inga teman angivna.")
    framsteg = framsteg or _tyst
    fran_ar = fran_ar or AR - 10
    d = DJUP[djup]
    exkl = {l.strip().upper() for l in exkludera if l.strip()}
    undantag = list(undantag)
    T = len(teman)

    # --- 1. sökning: tar fram namn
    s1, namn, positiva = samla_kandidater(oa, teman, fran_ar, d, framsteg)
    if not s1:
        raise IngaTraffar("Inga träffar. Prova bredare teman eller ett tidigare startår.")
    total = {aid: sum(per_tema.values()) for aid, per_tema in s1.items()}
    steg1_rang = {aid: i for i, aid in enumerate(sorted(total, key=lambda a: (-total[a], a)), 1)}
    ordning = valj_for_profil(s1, d["profiler"], T)

    framsteg(f"Hämtar profiler för {len(ordning)} forskare …", None)
    profiler = {kort_id(p["id"]): p for p in oa.hamta_forfattare(ordning, FORF_SELECT)}
    bortfall: dict[str, str] = {}
    kvar: list[str] = []
    for aid in ordning:
        prof = profiler.get(aid)
        if not prof:
            bortfall[aid] = "profilen saknas i OpenAlex"
            continue
        lander = {(i.get("country_code") or "").upper()
                  for i in prof.get("last_known_institutions") or [] if i} - {""}
        if lander and not lander & EUROPA:
            bortfall[aid] = f"lärosäte utanför Europa ({', '.join(sorted(lander))})"
        elif lander and lander <= exkl:
            bortfall[aid] = f"uteslutet land ({', '.join(sorted(lander))})"
        else:
            kvar.append(aid)

    # --- 2. profilkontroll: bedömer personen
    verkdata = profil.hamta_verk(oa, kvar, fran_ar, framsteg)
    kand: dict[str, dict] = {}
    for aid in kvar:
        data = verkdata.get(aid)
        if data is None:
            bortfall[aid] = "publikationerna kunde inte hämtas"
            continue
        verk = profil.rensa_verk(data["verk"])
        inst, lander = hemvist(profiler[aid], verk, exkl)
        if inst["land"] not in EUROPA:
            bortfall[aid] = f"lärosäte utanför Europa ({inst['land'] or 'okänt'})"
        elif inst["land"] in exkl:
            bortfall[aid] = f"uteslutet land ({inst['land']})"
        else:
            kand[aid] = {"verk": verk, "totalt": data.get("totalt") or len(verk),
                         "inst": inst, "lander": lander}
    if not kand:
        raise IngaTraffar("Ingen av de funna forskarna finns vid ett europeiskt lärosäte "
                          "som inte är uteslutet.")

    # Undantagen söks bara för att få exempel på hur sådan forskning ser ut.
    # Träffarna ger aldrig några kandidater.
    undantag_exempel: list[list[str]] = []
    for u in undantag:
        framsteg(f"Hämtar exempel på undantaget – {kort_etikett(u.etikett, 50)}", None)
        try:
            traffar = oa.semantisk_sokning(
                u.soktext, f"publication_year:>{fran_ar - 1},{TYPFILTER}",
                "id,title,relevance_score,abstract_inverted_index", 50)
        except RuntimeError as fel:
            framsteg(f"    inga exempel kunde hämtas: {fel}")
            traffar = []
        exempel = [_verktext(v) for v in traffar if profil.titelvikt(v.get("title") or "")][:25]
        undantag_exempel.append(exempel if len(exempel) >= 5 else [])

    fragor = [t.fulltext for t in teman] + [u.fulltext for u in undantag]
    exempel = [positiva.get(i, []) for i in range(T)] + undantag_exempel
    verk_per = {aid: k["verk"] for aid, k in kand.items()}
    jamforelse = profil.likhet(fragor, verk_per, exempel, likhet, oa.cache_dir, framsteg)
    likhetsmetod = jamforelse["metod"]
    G, _ = profil.kalibrera(jamforelse["S"][:T], jamforelse["P"][:T])
    undantagna = profil.undantagsverk(jamforelse["prototyp"], jamforelse["exempelniva"], T)
    aktiva_undantag = [u for j, u in enumerate(undantag) if undantag_exempel[j]]

    alla: list[dict] = []
    start = 0
    for aid, k in kand.items():
        n = len(k["verk"])
        b = profil.bedom(k["verk"], G[:, start:start + n], undantagna[:, start:start + n], exkl)
        start += n
        prof = profiler[aid]
        undantag_andel = b["undantag_andel"] if aktiva_undantag else None
        alla.append({
            "id": aid,
            "namn": prof.get("display_name") or namn.get(aid, ""),
            "larosate": k["inst"]["namn"],
            "land": k["inst"]["land"],
            "poang": matchning(b["styrka"], undantag_andel),
            "temastyrka": {t: s for t, s in enumerate(b["styrka"]) if s > 0.005},
            "tema_antal": {t: a for t, a in enumerate(b["antal"]) if a},
            "relevanta": b["relevanta"],
            "bedomda_verk": n,
            "verk_i_perioden": k["totalt"],
            "undantag_andel": undantag_andel,
            "andel_exkl": b["andel_exkl"],
            "sista5": b["sista5"],
            "medf": b["medf"],
            "toppverk": b["toppverk"],
            "h_index": (prof.get("summary_stats") or {}).get("h_index"),
            "antal_verk": prof.get("works_count"),
            "amnen": [t.get("display_name", "") for t in (prof.get("topics") or [])[:3]],
            "openalex_url": f"https://openalex.org/{aid}",
            "orcid": prof.get("orcid") or "",
            "ovriga_exkl": sorted(k["lander"] & exkl),
            "steg1_rang": steg1_rang[aid],
        })
    alla.sort(key=lambda k: (-k["poang"], k["steg1_rang"]))
    for i, k in enumerate(alla, 1):
        k["rang"] = i
    lista = alla[:antal_kandidater]

    # --- 3. senioritet för dem som listas
    forsta_ar = profil.parallellt(oa.forsta_publiceringsar, [k["id"] for k in lista],
                                  "Uppskattar akademisk ålder", framsteg)
    for k in lista:
        forsta = forsta_ar.get(k["id"])
        k["forsta_ar"] = forsta
        k["akad_alder"] = (AR - forsta) if forsta else None
        k["senioritet"] = senioritet(k["akad_alder"], k["h_index"], k["antal_verk"], k["sista5"])
        varn = profilvarning(k["akad_alder"], k["antal_verk"])
        if k["relevanta"] < MIN_RELEVANTA:
            varn.append(f"bara {k['relevanta']} relevanta verk")
        if (k["undantag_andel"] or 0) >= 0.25:
            varn.append(f"{round(100 * k['undantag_andel'])} % av verken ligger i undantagna områden")
        if k["ovriga_exkl"]:
            varn.append("även verksam i " + ", ".join(k["ovriga_exkl"]))
        if (k["andel_exkl"] or 0) >= 0.25:
            varn.append(f"{round(100 * k['andel_exkl'])} % av verken har medförfattare i "
                        + ", ".join(sorted(exkl)) + " – jävsrisk")
        k["varning"] = "; ".join(varn)
        k["valbar"] = (k["senioritet"] in ("Trolig", "Möjlig") and k["relevanta"] >= MIN_RELEVANTA
                       and not k["ovriga_exkl"] and (k["undantag_andel"] or 0) < MAX_UNDANTAG)

    # --- 4. gruppförslag
    grupp = foresla_grupp(lista, T, gruppstorlek, max_per_land)
    reserver = valj_reserver(grupp, lista, max_per_land)

    return {
        "kandidater": lista,
        "grupp": grupp,
        "reserver": reserver,
        "teman": teman,
        "undantag": undantag,
        "tunt": tunt_tackta(grupp, T),
        "tratt": {
            "steg1": steg1_rang,
            "bortfall": bortfall,
            "profilerade": {k["id"]: k["rang"] for k in alla},
            "orcid": {k["orcid"].rsplit("/", 1)[-1]: k["id"] for k in alla if k["orcid"]},
            "namn": {**namn, **{k["id"]: k["namn"] for k in alla}},
        },
        "metod": {
            "fran_ar": fran_ar, "djup": djup, "exkludera": sorted(exkl), "likhet": likhetsmetod,
            "antal_steg1": len(s1), "antal_profilerade": len(ordning), "antal_bedomda": len(alla),
            "antal_valbara": sum(1 for k in lista if k["valbar"]),
            "inaktiva_undantag": [u.etikett for u in undantag if u not in aktiva_undantag],
            "anrop": oa.antal_anrop, "kostnad": oa.kostnad_usd, "kvar": oa.kvar_usd,
            "gruppstorlek": gruppstorlek, "max_per_land": max_per_land,
        },
    }


# ---------------------------------------------------------------- Excel
RUBRIK_FONT = Font(bold=True, color="FFFFFF")
RUBRIK_FYLL = PatternFill("solid", fgColor="2F4858")
FARG = {"Trolig": "D8F0DC", "Möjlig": "FFF2CC", "Osäker/yngre": "F4D6D6"}
LANKKOLUMNER = ("OpenAlex-profil", "ORCID")

KAND_KOLUMNER = [
    ("Rang", 6), ("Namn", 24), ("Lärosäte (senast kända)", 30), ("Land", 6),
    ("Matchning", 10), ("Relevanta verk", 9), ("Docentnivå?", 12), ("h-index", 8),
    ("Antal verk", 9), ("Första publ.år", 10), ("Akad. ålder", 9),
    ("Sista-/ensamförf. 5 år", 11), ("Starkaste teman (antal verk)", 38),
    ("Mest relevanta publikationer", 60), ("Andel undantag", 9),
    ("Medförf. i uteslutna länder", 11), ("Huvudämnen (OpenAlex)", 34),
    ("Varning", 30), ("OpenAlex-profil", 30), ("ORCID", 30),
]
GRUPP_KOLUMNER = KAND_KOLUMNER[:4] + [("Reserver", 34)] + KAND_KOLUMNER[4:]


def _procent(andel: float | None) -> str:
    return "" if andel is None else f"{round(100 * andel)} %"


def _komma(tal: float, decimaler: int | None = None) -> str:
    """Tal med decimalkomma."""
    return (f"{tal:g}" if decimaler is None else f"{tal:.{decimaler}f}").replace(".", ",")


def kandidatrad(k: dict, teman: list[Tema]) -> list:
    starka = sorted(((s, t) for t, s in k["temastyrka"].items() if s >= 0.3), reverse=True)
    return [
        k["rang"], k["namn"], k["larosate"], k["land"], round(k["poang"], 1), k["relevanta"],
        k["senioritet"], k["h_index"], k["antal_verk"], k["forsta_ar"], k["akad_alder"], k["sista5"],
        ", ".join(f"{kort_etikett(teman[t].etikett, 40)} ({k['tema_antal'].get(t, 0)})"
                  for _, t in starka),
        "\n".join(f"{titel} ({ar})" for _, titel, ar, _ in k["toppverk"]),
        _procent(k["undantag_andel"]), _procent(k["andel_exkl"]),
        ", ".join(k["amnen"]), k["varning"], k["openalex_url"], k["orcid"],
    ]


def grupprad(k: dict, teman: list[Tema], reserver: list[dict]) -> list:
    rad = kandidatrad(k, teman)
    return rad[:4] + ["\n".join(f"{r['namn']} (rang {r['rang']}, {r['land']})" for r in reserver)] + rad[4:]


def skriv_blad(ws, kolumner: list[tuple[str, int]], rader: list[list]):
    rubriker = [namn for namn, _ in kolumner]
    lankar = {c for c, namn in enumerate(rubriker, 1) if namn in LANKKOLUMNER}
    senior = rubriker.index("Docentnivå?") + 1 if "Docentnivå?" in rubriker else None
    for c, (namn, bredd) in enumerate(kolumner, 1):
        cell = ws.cell(row=1, column=c, value=namn)
        cell.font, cell.fill = RUBRIK_FONT, RUBRIK_FYLL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(c)].width = bredd
    for r, rad in enumerate(rader, 2):
        for c, varde in enumerate(rad, 1):
            cell = ws.cell(row=r, column=c, value=varde)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if c in lankar and varde:
                cell.hyperlink = varde
                cell.style = "Hyperlink"
                cell.alignment = Alignment(vertical="top")
        if senior and rad[senior - 1] in FARG:
            ws.cell(row=r, column=senior).fill = PatternFill("solid", fgColor=FARG[rad[senior - 1]])
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions


def tematackning(res: dict) -> list[list]:
    """Rader till bladet Per tema: tema, vilka i gruppen som täcker det och de
    starkaste kandidaterna."""
    rader = []
    for t, tema in enumerate(res["teman"]):
        tacker = [k for k in res["grupp"] if k["temastyrka"].get(t, 0.0) >= TACKT]
        text = ", ".join(f"{k['namn']} ({k['tema_antal'].get(t, 0)})" for k in tacker)
        if not tacker:
            text = "– EJ TÄCKT –"
        elif len(tacker) < 2:
            text = "TUNT TÄCKT (en person): " + text
        basta = sorted((k for k in res["kandidater"] if k["temastyrka"].get(t, 0.0) > 0),
                       key=lambda k: -k["temastyrka"][t])[:10]
        rader.append([kort_etikett(tema.etikett, 120), text,
                      "\n".join(f"{k['namn']} ({k['land']}, {k['senioritet']}, "
                                f"{k['tema_antal'].get(t, 0)} verk)" for k in basta)])
    return rader


def metodtext(res: dict) -> list[str]:
    m = res["metod"]
    undantag = res["undantag"]
    return [
        "Så togs listan fram",
        f"Körd: {dt.datetime.now():%Y-%m-%d %H:%M}. Källa: OpenAlex (api.openalex.org). "
        f"Nya API-anrop: {m['anrop']}, kostnad {_komma(m['kostnad'], 3)} dollar. Övriga svar kom från cachen.",
        f"Publikationer från {m['fran_ar']} och framåt. Uteslutna länder: "
        f"{', '.join(m['exkludera']) or 'inga'}. Sökdjup: {m['djup']}.",
        "",
        "1. Sökning – tar fram namn",
        "  Varje tema söktes semantiskt (på beskrivningen) och med nyckelord (på sökfrasen, bara Europa).",
        f"  Sökningen gav {m['antal_steg1']} författare med europeisk affiliering. De {m['antal_profilerade']} "
        "med flest och bäst träffar gick vidare, med minst några för varje tema.",
        "",
        "2. Profilkontroll – bedömer personen",
        "  För varje forskare hämtades de upp till 100 senaste publikationerna i perioden.",
        f"  Varje publikation jämfördes med varje tema. Jämförelsen gjordes lokalt på datorn med {m['likhet']}.",
        "  Relevans mäts på en skala 0–1: 0 är en typisk publikation bland kandidaterna, 1 är temats bästa sökträffar.",
        f"  Ett verk räknas som relevant från {_komma(profil.RELEVANT)}. Verk under {_komma(profil.GOLV)} bidrar inte alls.",
        "  Temastyrka (0–1) växer med antalet relevanta verk: fem fullt relevanta verk ger 0,5, tio ger 0,76 och tjugo 0,94.",
        "  Första-, sista- och korresponderande författare väger fullt. Mittenförfattare väger mindre ju fler författarna är.",
        "  Verk äldre än fem år väger 0,75. Protokoll, ledare och prioriteringsartiklar väger 0,5.",
        "  Matchning (0–100) = starkaste temat + en tredjedel av näst starkaste + en sjättedel av det tredje.",
        "  En forskare som är mycket stark i ett enda tema får alltså som mest 67. Se bladet Per tema för de starkaste per tema.",
        f"  {m['antal_bedomda']} forskare bedömdes. Övriga föll bort på land eller saknade data.",
        ("  Undantag: ett verk räknas till ett undantaget område när det liknar exempel på sådan forskning tydligt mer än "
         f"det liknar något tema. Andelen sådana verk sänker matchningen med upp till {round(100 * UNDANTAGSAVDRAG)} %. "
         "Det fungerar bäst för tydligt avgränsade områden, som läkemedelsstudier." if undantag else
         "  Inga undantag angavs."),
        *(["  Dessa undantag kunde inte användas (för få sökträffar): " + "; ".join(m["inaktiva_undantag"])]
          if m["inaktiva_undantag"] else []),
        "",
        "Docentnivå? – grov uppskattning, INTE verifierad titel:",
        f"  Trolig: akademisk ålder ≥ {TROLIG['akademisk_alder']} år, h-index ≥ {TROLIG['h_index']}, "
        f"≥ {TROLIG['antal_verk']} verk och minst {TROLIG['sista_forfattare']} verk som sista- eller "
        "ensamförfattare de senaste fem åren.",
        f"  Möjlig: akademisk ålder ≥ {MOJLIG['akademisk_alder']} år, h-index ≥ {MOJLIG['h_index']}, "
        f"≥ {MOJLIG['antal_verk']} verk, eller Trolig-trösklarna utan sistaförfattarskap.",
        "  Akademisk ålder = år sedan första publikation i OpenAlex. Tröskelvärdena kan ändras överst i rekrytera.py.",
        "  I ämnen där författarna står i bokstavsordning säger sistaförfattarskap mindre.",
        "  Docent motsvaras utomlands ofta av associate professor/Reader/habilitation – kontrollera på lärosätets webbsida.",
        "",
        "Gruppförslaget",
        f"  Bara personer med minst {MIN_RELEVANTA} relevanta verk och docentnivå Trolig eller Möjlig kan föreslås. "
        f"{m['antal_valbara']} av de listade uppfyller det.",
        f"  Den som också är verksam i ett uteslutet land, eller har minst {round(100 * MAX_UNDANTAG)} % av verken "
        "i undantagna områden, föreslås inte men står kvar i kandidatlistan.",
        f"  Gruppen byggs så att varje tema helst täcks av två personer, med högst {m['max_per_land']} per land.",
        "  Avdrag ges för sampublicering med någon som redan valts, för samma lärosäte och för många "
        "medförfattare i de uteslutna länderna.",
        "  Reserver: för varje plats två personer med liknande temaprofil som inte har publicerat med "
        "någon av de övriga i gruppen.",
        "",
        "Kända begränsningar",
        "  • OpenAlex kan slå ihop eller dela upp personer med samma namn – kontrollera profilen (se kolumnen Varning).",
        "  • 'Lärosäte (senast kända)' kan vara inaktuellt om personen bytt arbetsplats nyligen.",
        "  • Forskare med fler än 100 verk i perioden bedöms på de 100 senaste.",
        "  • Kön, språkkunskaper och tillgänglighet finns inte i datan och måste hanteras manuellt.",
        "  • Jäv mot sökande kan bara kontrolleras när ansökningarna kommit in (se fordela.py).",
        "",
        "Nästa steg: markera med x i kolumnen 'Vald' på bladet Förslag grupp eller Kandidater och spara.",
        "Filen kan sedan användas direkt som granskarlista i fordela.py.",
        "",
        "Teman som användes:",
        *[f"  {i + 1}. {t.fras or '(ingen sökfras)'} | {kort_etikett(t.beskrivning, 220)}"
          for i, t in enumerate(res["teman"])],
        *(["", "Undantag:"] + [f"  – {u.fras or '(ingen fras)'} | {kort_etikett(u.beskrivning, 220)}"
                               for u in undantag] if undantag else []),
    ]


FETA_RADER = {"Så togs listan fram", "1. Sökning – tar fram namn", "2. Profilkontroll – bedömer personen",
              "Gruppförslaget", "Kända begränsningar", "Teman som användes:", "Undantag:"}


def skriv_excel(ut, res: dict):
    """Skriver resultatet till en Excel-fil (sökväg eller filobjekt)."""
    teman = res["teman"]
    wb = Workbook()

    ws = wb.active
    ws.title = "Förslag grupp"
    skriv_blad(ws, [("Vald (x)", 8)] + GRUPP_KOLUMNER,
               [[""] + grupprad(k, teman, res["reserver"].get(k["id"], [])) for k in res["grupp"]])

    ws = wb.create_sheet("Kandidater")
    skriv_blad(ws, [("Vald (x)", 8)] + KAND_KOLUMNER,
               [[""] + kandidatrad(k, teman) for k in res["kandidater"]])

    ws = wb.create_sheet("Per tema")
    skriv_blad(ws, [("Tema", 45), ("Täcks i gruppförslaget av (antal relevanta verk)", 45),
                    ("Starkaste kandidaterna", 55)], tematackning(res))

    ws = wb.create_sheet("Metod")
    for r, rad in enumerate(metodtext(res), 1):
        cell = ws.cell(row=r, column=1, value=rad)
        if rad in FETA_RADER:
            cell.font = Font(bold=True, size=12 if r == 1 else 11)
    ws.column_dimensions["A"].width = 140

    wb.save(ut)


# ---------------------------------------------------------------- huvudprogram
def main(argv=None):
    p = argparse.ArgumentParser(description="Hitta europeiska granskare för en utlysning via OpenAlex.")
    p.add_argument("--teman", type=Path,
                   help="Temafil: en rad per tema, 'kort sökfras | beskrivning' (se exempel/teman.txt)")
    p.add_argument("--utlysning", type=Path, help="Utlysningstexten (.txt). Behövs bara för --temaprompt")
    p.add_argument("--temaprompt", action="store_true",
                   help="Skriv ut en färdig fråga till en AI-assistent som ger ett temaförslag, och avsluta")
    p.add_argument("--fran-ar", type=int, default=AR - 10,
                   help="Räkna publikationer från detta år (standard: 10 år bakåt)")
    p.add_argument("--djup", choices=list(DJUP), default="normal",
                   help="Sökdjup: snabb (billigast), normal eller grundlig")
    p.add_argument("--antal-kandidater", type=int, default=150, help="Hur många som listas")
    p.add_argument("--exkludera-lander", default="", help="Kommaseparerade landskoder att utesluta, t.ex. SE")
    p.add_argument("--gruppstorlek", type=int, default=10)
    p.add_argument("--max-per-land", type=int, default=2)
    p.add_argument("--likhet", choices=["auto", "modell", "tfidf"], default="auto",
                   help="Profilkontrollens jämförelse: språkmodell om den finns (auto), "
                        "tvinga språkmodell (modell) eller ordjämförelse (tfidf)")
    p.add_argument("--ut", type=Path, default=Path("kandidater.xlsx"))
    args = p.parse_args(argv)
    exkludera = [l.strip().upper() for l in args.exkludera_lander.split(",") if l.strip()]

    if args.temaprompt:
        if not args.utlysning:
            raise SystemExit("--temaprompt behöver utlysningen: --utlysning FIL --temaprompt")
        sys.stdout.reconfigure(encoding="utf-8")
        print(ai_prompt(args.utlysning.read_text(encoding="utf-8-sig")))
        return
    if not args.teman:
        raise SystemExit(
            "Ange en temafil med --teman (se exempel/teman.txt). Teman är de kompetenser gruppen behöver.\n"
            "Få ett förslag från en AI-assistent: python rekrytera.py --utlysning FIL --temaprompt")
    teman, undantag = las_temafil(args.teman)
    if not teman:
        raise SystemExit(f"Hittade inga teman i {args.teman}.")
    for varning in granska_teman(teman, undantag):
        print(f"OBS: {varning}", file=sys.stderr)

    def framsteg(text: str, andel: float | None = None):
        print(f"  {text}", file=sys.stderr)

    oa = OpenAlex()
    kostnad = uppskatta_kostnad(len(teman), len(undantag), args.djup, args.antal_kandidater)
    kvar = oa.kolla_budget()
    print(f"Söker med {len(teman)} teman och {len(undantag)} undantag, sökdjup {args.djup}. "
          f"Kostar högst cirka {_komma(kostnad, 2)} dollar"
          + (f" (kvar av dagsbudgeten: {_komma(kvar, 2)})." if kvar is not None else "."), file=sys.stderr)
    if kvar is not None and kvar < kostnad:
        print("OBS: budgeten kan ta slut under körningen. Det som hunnit hämtas sparas i cachen. "
              "Välj --djup snabb eller skaffa en gratis API-nyckel (1 dollar per dag).", file=sys.stderr)

    try:
        res = kor(oa, teman, undantag, fran_ar=args.fran_ar, djup=args.djup,
                  antal_kandidater=args.antal_kandidater, exkludera=exkludera,
                  gruppstorlek=args.gruppstorlek, max_per_land=args.max_per_land,
                  likhet=args.likhet, framsteg=framsteg)
    except IngaTraffar as fel:
        raise SystemExit(str(fel))

    ut = args.ut
    try:
        skriv_excel(ut, res)
    except PermissionError:     # filen är öppen i Excel
        ut = ut.with_name(f"{ut.stem}_{dt.datetime.now():%H%M%S}{ut.suffix}")
        skriv_excel(ut, res)
        print(f"OBS: {args.ut} är öppen i ett annat program – sparade till {ut} i stället.", file=sys.stderr)

    m = res["metod"]
    print(f"\nKlart: {len(res['kandidater'])} kandidater ({m['antal_valbara']} valbara till gruppen), "
          f"gruppförslag med {len(res['grupp'])} personer → {ut}", file=sys.stderr)
    if len(res["grupp"]) < args.gruppstorlek:
        print(f"Bara {len(res['grupp'])} av {args.gruppstorlek} platser kunde fyllas. Prova --djup grundlig, "
              "fler länder per plats eller bredare teman.", file=sys.stderr)
    for t, n in res["tunt"]:
        print(f"Tunt täckt tema ({n} person{'er' if n != 1 else ''} i gruppen): "
              f"{kort_etikett(teman[t].etikett, 60)}", file=sys.stderr)
    print(f"Profilkontroll: {m['likhet']}. API-anrop denna körning: {m['anrop']} "
          f"(kostnad {_komma(m['kostnad'], 3)} dollar, resten från cache).", file=sys.stderr)


if __name__ == "__main__":
    main()
