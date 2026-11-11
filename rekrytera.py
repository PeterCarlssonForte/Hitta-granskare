"""Steg 1: hitta europeiska forskare som passar en utlysning.

Exempel:
    python rekrytera.py --utlysning exempel/utlysning.txt --teman exempel/teman.txt \
        --exkludera-lander SE --gruppstorlek 10 --ut kandidater.xlsx

Flöde
 1. Varje tema (en rad i teman-filen) och utlysningstexten används som sökfrågor
    mot OpenAlex: semantisk sökning (betydelse) + nyckelordssökning (exakta termer),
    begränsat till publikationer med minst en europeisk affiliering.
 2. Författare på träffarna poängsätts efter hur relevanta, hur nya och hur
    centrala (första/sista författare) deras träffar är, och hur många teman de täcker.
 3. De bästa kandidaterna profileras: aktuellt lärosäte och land, h-index,
    antal publikationer och första publiceringsår (akademisk ålder).
 4. En grov senioritetsbedömning görs (docentnivå trolig/möjlig/osäker).
    OBS: docentur finns inte som data – bedömningen måste verifieras.
 5. Ett förslag på beredningsgrupp tas fram som täcker alla teman med
    spridning över länder och lärosäten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openalex import EUROPA, OpenAlex, kort_id

AR = dt.date.today().year

# Trösklar för senioritetsbedömningen. Justera efter ämnesområde – t.ex. har
# samhällsvetare ofta lägre h-index än medicinare på samma karriärnivå.
TROLIG = {"akademisk_alder": 12, "h_index": 12, "antal_verk": 30}
MOJLIG = {"akademisk_alder": 8, "h_index": 8, "antal_verk": 15}

VERK_SELECT = "id,title,publication_year,authorships,relevance_score,doi"
FORF_SELECT = ("id,display_name,orcid,works_count,cited_by_count,summary_stats,"
               "last_known_institutions,topics")


# ---------------------------------------------------------------- inläsning
def las_teman(utlysning: str, temafil: Path | None) -> list[str]:
    if temafil:
        rader = [r.strip(" -•\t") for r in temafil.read_text(encoding="utf-8").splitlines()]
        teman = [r for r in rader if r and not r.startswith("#")]
        if teman:
            return teman
    # Utan temafil: dela utlysningen i stycken om högst ~1500 tecken.
    stycken = [s.strip() for s in re.split(r"\n\s*\n", utlysning) if len(s.strip()) > 80]
    teman, buffert = [], ""
    for s in stycken:
        if len(buffert) + len(s) > 1500 and buffert:
            teman.append(buffert)
            buffert = ""
        buffert = (buffert + "\n" + s).strip()
    if buffert:
        teman.append(buffert)
    return teman or [utlysning[:2000]]


def ser_svensk_ut(text: str) -> bool:
    ord_ = re.findall(r"\b\w+\b", text.lower())
    svenska = {"och", "att", "som", "för", "med", "det", "inom", "av", "på", "är"}
    return sum(o in svenska for o in ord_) > max(5, len(ord_) * 0.05)


def kort_etikett(tema: str, n: int = 60) -> str:
    rad = tema.splitlines()[0]
    return rad if len(rad) <= n else rad[: n - 1] + "…"


# ---------------------------------------------------------------- poängsättning
def verk_vikt(verk: dict, rang: int, max_rel: float) -> float:
    rel = verk.get("relevance_score")
    if rel and max_rel:
        bas = rel / max_rel
    else:
        bas = 1 / (1 + rang / 10)
    ar = verk.get("publication_year") or 0
    aktualitet = 1.0 if ar >= AR - 5 else 0.75 if ar >= AR - 10 else 0.5
    return bas * aktualitet


def europeiska_lander(authorship: dict) -> set[str]:
    lander = set(authorship.get("countries") or [])
    for inst in authorship.get("institutions") or []:
        if inst.get("country_code"):
            lander.add(inst["country_code"])
    return {l.upper() for l in lander} & EUROPA


def samla_kandidater(oa: OpenAlex, teman: list[str], utlysning: str, fran_ar: int,
                     per_sokning: int) -> tuple[dict, dict]:
    filter_ = (f"authorships.institutions.continent:europe,"
               f"publication_year:>{fran_ar - 1},type:article|review|book|book-chapter")
    # Semantisk sökning stöder inte continent-filtret – europafiltrering sker efteråt.
    semantisk_filter_ = f"publication_year:>{fran_ar - 1},type:article|review|book|book-chapter"
    poang: dict[str, float] = defaultdict(float)
    info: dict[str, dict] = defaultdict(lambda: {
        "namn": "", "teman": defaultdict(float), "verk": {}})

    fragor: list[tuple[int | None, str, str]] = []
    for i, tema in enumerate(teman):
        fragor.append((i, "semantisk", tema))
        if len(tema) <= 200:            # nyckelordssökning bara för korta fraser
            fragor.append((i, "nyckelord", tema))
    if len(teman) > 1 or teman[0] != utlysning[:2000]:
        fragor.append((None, "semantisk", utlysning[:2000]))

    for nr, (tema_idx, typ, text) in enumerate(fragor, 1):
        etikett = "hela utlysningen" if tema_idx is None else kort_etikett(teman[tema_idx], 50)
        print(f"  [{nr}/{len(fragor)}] {typ:9s} – {etikett}", file=sys.stderr)
        try:
            if typ == "semantisk":
                traffar = oa.semantisk_sokning(text, semantisk_filter_, VERK_SELECT, per_sokning)
            else:
                traffar = oa.nyckelordssokning(text, filter_, VERK_SELECT, per_sokning)
        except RuntimeError as fel:
            print(f"    hoppar över frågan: {fel}", file=sys.stderr)
            continue
        max_rel = max((v.get("relevance_score") or 0 for v in traffar), default=0)
        for rang, verk in enumerate(traffar):
            vikt = verk_vikt(verk, rang, max_rel)
            for au in verk.get("authorships") or []:
                aid = (au.get("author") or {}).get("id")
                if not aid or not europeiska_lander(au):
                    continue
                pos = au.get("author_position")
                v = vikt * (1.0 if pos in ("first", "last") else 0.6)
                poang[aid] += v
                post = info[aid]
                post["namn"] = (au.get("author") or {}).get("display_name", "")
                if tema_idx is not None:
                    post["teman"][tema_idx] = max(post["teman"][tema_idx], v)
                tidigare = post["verk"].get(verk["id"])
                if tidigare is None or tidigare[0] < v:
                    post["verk"][verk["id"]] = (v, verk.get("title") or "",
                                                verk.get("publication_year"), verk.get("doi"))

    # Bonus för bredd: forskare som träffar flera teman väger tyngre.
    for aid in poang:
        antal_teman = len(info[aid]["teman"])
        poang[aid] *= 1 + 0.25 * max(0, antal_teman - 1)
        # Dämpa effekten av att en person bara har många halvbra träffar.
        poang[aid] = math.sqrt(poang[aid]) * math.log1p(len(info[aid]["verk"]))
    return poang, info


# ---------------------------------------------------------------- senioritet
def senioritet(akad_alder: int | None, h: int | None, verk: int | None) -> str:
    a, h, w = akad_alder or 0, h or 0, verk or 0
    if a >= TROLIG["akademisk_alder"] and h >= TROLIG["h_index"] and w >= TROLIG["antal_verk"]:
        return "Trolig"
    if a >= MOJLIG["akademisk_alder"] and h >= MOJLIG["h_index"] and w >= MOJLIG["antal_verk"]:
        return "Möjlig"
    return "Osäker/yngre"


def profilvarning(akad_alder: int | None, verk: int | None) -> str:
    varn = []
    if akad_alder and akad_alder > 50:
        varn.append("mycket lång publiceringshistorik – profilen kan vara sammanslagen")
    if verk and verk > 1500:
        varn.append("ovanligt många verk – kontrollera profilen")
    return "; ".join(varn)


# ---------------------------------------------------------------- gruppförslag
def foresla_grupp(kandidater: list[dict], antal_teman: int, storlek: int,
                  max_per_land: int) -> list[dict]:
    valbara = [k for k in kandidater if k["senioritet"] in ("Trolig", "Möjlig")]
    if not valbara:
        return []
    maxpoang = max(k["poang"] for k in valbara) or 1
    valda: list[dict] = []
    tackning = [0.0] * antal_teman
    lander: Counter = Counter()
    larosaten: Counter = Counter()
    while len(valda) < storlek:
        bast, bast_varde = None, -1e9
        for k in valbara:
            if k in valda or lander[k["land"]] >= max_per_land:
                continue
            varde = k["poang"] / maxpoang
            # Vinst i tematäckning: hur mycket kandidaten lyfter svagt täckta teman.
            for t, styrka in k["temastyrka"].items():
                varde += 1.5 * max(0.0, min(1.0, styrka) - tackning[t])
            if k["senioritet"] == "Möjlig":
                varde -= 0.15
            if larosaten[k["larosate"]]:
                varde -= 0.5
            if varde > bast_varde:
                bast, bast_varde = k, varde
        if bast is None:
            break
        valda.append(bast)
        lander[bast["land"]] += 1
        larosaten[bast["larosate"]] += 1
        for t, styrka in bast["temastyrka"].items():
            tackning[t] = max(tackning[t], min(1.0, styrka))
    return valda


# ---------------------------------------------------------------- Excel
RUBRIK_FONT = Font(bold=True, color="FFFFFF")
RUBRIK_FYLL = PatternFill("solid", fgColor="2F4858")
FARG = {"Trolig": "D8F0DC", "Möjlig": "FFF2CC", "Osäker/yngre": "F4D6D6"}


def skriv_blad(ws, kolumner: list[tuple[str, int]], rader: list[list], lank_kol: dict | None = None,
               senior_kol: int | None = None):
    for c, (namn, bredd) in enumerate(kolumner, 1):
        cell = ws.cell(row=1, column=c, value=namn)
        cell.font, cell.fill = RUBRIK_FONT, RUBRIK_FYLL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(c)].width = bredd
    for r, rad in enumerate(rader, 2):
        for c, varde in enumerate(rad, 1):
            cell = ws.cell(row=r, column=c, value=varde)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if lank_kol and c in lank_kol and varde:
                cell.hyperlink = varde
                cell.style = "Hyperlink"
                cell.alignment = Alignment(vertical="top")
        if senior_kol:
            s = rad[senior_kol - 1]
            if s in FARG:
                ws.cell(row=r, column=senior_kol).fill = PatternFill("solid", fgColor=FARG[s])
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions


def kandidatrad(k: dict, teman: list[str], med_vald: bool = False) -> list:
    rad = [
        k["rang"], k["namn"], k["larosate"], k["land"], round(k["poang"], 2),
        k["senioritet"], k["h_index"], k["antal_verk"], k["forsta_ar"], k["akad_alder"],
        ", ".join(kort_etikett(teman[t], 40) for t, s in
                  sorted(k["temastyrka"].items(), key=lambda x: -x[1]) if s >= 0.3),
        "\n".join(f"{t} ({a})" for _, t, a, _ in k["toppverk"]),
        ", ".join(k["amnen"]), k["varning"], k["openalex_url"], k["orcid"],
    ]
    return (["" ] + rad) if med_vald else rad


KAND_KOLUMNER = [
    ("Rang", 6), ("Namn", 24), ("Lärosäte (senast kända)", 30), ("Land", 6),
    ("Matchning", 10), ("Docentnivå?", 12), ("h-index", 8), ("Antal verk", 9),
    ("Första publ.år", 10), ("Akad. ålder", 9), ("Täckta teman", 34),
    ("Mest relevanta publikationer", 60), ("Huvudämnen (OpenAlex)", 34),
    ("Varning", 24), ("OpenAlex-profil", 30), ("ORCID", 30),
]


def skriv_excel(ut: Path, kandidater: list[dict], grupp: list[dict], teman: list[str],
                args, antal_anrop: int):
    wb = Workbook()

    ws = wb.active
    ws.title = "Förslag grupp"
    skriv_blad(ws, [("Vald (x)", 8)] + KAND_KOLUMNER,
               [kandidatrad(k, teman, med_vald=True) for k in grupp],
               lank_kol={16, 17}, senior_kol=7)

    ws = wb.create_sheet("Kandidater")
    skriv_blad(ws, [("Vald (x)", 8)] + KAND_KOLUMNER,
               [kandidatrad(k, teman, med_vald=True) for k in kandidater],
               lank_kol={16, 17}, senior_kol=7)

    ws = wb.create_sheet("Per tema")
    rader = []
    for t, tema in enumerate(teman):
        basta = sorted((k for k in kandidater if k["temastyrka"].get(t, 0) > 0),
                       key=lambda k: -k["temastyrka"][t])[:10]
        i_grupp = [k["namn"] for k in grupp if k["temastyrka"].get(t, 0) >= 0.3]
        rader.append([kort_etikett(tema, 120), ", ".join(i_grupp) or "– EJ TÄCKT –",
                      "\n".join(f"{k['namn']} ({k['land']}, {k['senioritet']})" for k in basta)])
    skriv_blad(ws, [("Tema", 45), ("Täcks i gruppförslaget av", 40),
                    ("Starkaste kandidaterna", 50)], rader)

    ws = wb.create_sheet("Metod")
    text = [
        "Så togs listan fram",
        f"Körd: {dt.datetime.now():%Y-%m-%d %H:%M}. Källa: OpenAlex (api.openalex.org), {antal_anrop} nya API-anrop.",
        f"Publikationer från {args.fran_ar} och framåt, minst en författare med europeisk affiliering.",
        f"Uteslutna länder: {', '.join(args.exkludera_lander) or 'inga'}.",
        "Varje tema söktes semantiskt (betydelse) och, för korta fraser, med nyckelord.",
        "Matchning = viktad summa av träffar (relevans × aktualitet × första/sista författare), med bonus för flera teman.",
        "",
        "Docentnivå? – grov uppskattning, INTE verifierad titel:",
        f"  Trolig: akademisk ålder ≥ {TROLIG['akademisk_alder']} år, h-index ≥ {TROLIG['h_index']}, ≥ {TROLIG['antal_verk']} verk.",
        f"  Möjlig: akademisk ålder ≥ {MOJLIG['akademisk_alder']} år, h-index ≥ {MOJLIG['h_index']}, ≥ {MOJLIG['antal_verk']} verk.",
        "  Akademisk ålder = år sedan första publikation i OpenAlex. Tröskelvärdena kan ändras överst i rekrytera.py.",
        "  Docent motsvaras utomlands ofta av associate professor/Reader/habilitation – kontrollera på lärosätets webbsida.",
        "",
        "Kända begränsningar",
        "  • OpenAlex kan slå ihop eller dela upp personer med samma namn – kontrollera profilen (se kolumnen Varning).",
        "  • 'Lärosäte (senast kända)' kan vara inaktuellt om personen bytt arbetsplats nyligen.",
        "  • Kön, språkkunskaper och tillgänglighet finns inte i datan och måste hanteras manuellt.",
        "  • Jäv mot sökande kan bara kontrolleras när ansökningarna kommit in (se fordela.py).",
        "",
        "Nästa steg: markera med x i kolumnen 'Vald' på bladet Förslag grupp eller Kandidater och spara.",
        "Filen kan sedan användas direkt som granskarlista i fordela.py.",
        "",
        "Teman som användes:",
    ] + [f"  {i + 1}. {kort_etikett(t, 150)}" for i, t in enumerate(teman)]
    for r, rad in enumerate(text, 1):
        cell = ws.cell(row=r, column=1, value=rad)
        if r == 1 or rad in ("Kända begränsningar", "Teman som användes:"):
            cell.font = Font(bold=True, size=12 if r == 1 else 11)
    ws.column_dimensions["A"].width = 130

    wb.save(ut)


# ---------------------------------------------------------------- huvudprogram
def main(argv=None):
    p = argparse.ArgumentParser(description="Hitta europeiska granskare för en utlysning via OpenAlex.")
    p.add_argument("--utlysning", type=Path, required=True, help="Utlysningstexten (.txt)")
    p.add_argument("--teman", type=Path, help="Valfri fil med ett tema per rad (helst på engelska)")
    p.add_argument("--fran-ar", type=int, default=AR - 10, help="Räkna publikationer från detta år (standard: 10 år bakåt)")
    p.add_argument("--per-sokning", type=int, default=50, help="Antal publikationer per sökfråga (semantisk max 50)")
    p.add_argument("--antal-kandidater", type=int, default=150, help="Hur många som profileras och listas")
    p.add_argument("--exkludera-lander", default="", help="Kommaseparerade landskoder att utesluta, t.ex. SE")
    p.add_argument("--gruppstorlek", type=int, default=10)
    p.add_argument("--max-per-land", type=int, default=2)
    p.add_argument("--ut", type=Path, default=Path("kandidater.xlsx"))
    args = p.parse_args(argv)
    args.exkludera_lander = [l.strip().upper() for l in args.exkludera_lander.split(",") if l.strip()]

    utlysning = args.utlysning.read_text(encoding="utf-8")
    teman = las_teman(utlysning, args.teman)
    if any(ser_svensk_ut(t) for t in teman):
        print("Tips: OpenAlex innehåll är till största delen på engelska. Teman skrivna på "
              "engelska ger oftast bättre träffar.", file=sys.stderr)

    oa = OpenAlex()
    print(f"Söker i OpenAlex med {len(teman)} teman …", file=sys.stderr)
    poang, info = samla_kandidater(oa, teman, utlysning, args.fran_ar, args.per_sokning)
    if not poang:
        raise SystemExit("Inga träffar. Prova bredare teman eller ett tidigare --fran-ar.")

    # Profilera fler än som behövs eftersom vissa faller bort på land/senioritet.
    topp = sorted(poang, key=lambda a: -poang[a])[: int(args.antal_kandidater * 1.5)]
    print(f"Hämtar profiler för {len(topp)} forskare …", file=sys.stderr)
    profiler = {p_["id"]: p_ for p_ in oa.hamta_forfattare(topp, FORF_SELECT)}

    kandidater = []
    for aid in topp:
        prof = profiler.get(aid)
        if not prof:
            continue
        inst = (prof.get("last_known_institutions") or [{}])[0] or {}
        land = (inst.get("country_code") or "").upper()
        if land not in EUROPA or land in args.exkludera_lander:
            continue
        kandidater.append({"id": aid, "prof": prof, "inst": inst, "land": land})
        if len(kandidater) >= args.antal_kandidater:
            break

    print(f"Uppskattar akademisk ålder för {len(kandidater)} kandidater …", file=sys.stderr)
    maxtema = defaultdict(float)
    for k in kandidater:
        for t, s in info[k["id"]]["teman"].items():
            maxtema[t] = max(maxtema[t], s)

    ut = []
    for k in kandidater:
        prof, aid = k["prof"], k["id"]
        forsta = oa.forsta_publiceringsar(aid)
        akad = (AR - forsta) if forsta else None
        h = (prof.get("summary_stats") or {}).get("h_index")
        verk = prof.get("works_count")
        toppverk = sorted(info[aid]["verk"].values(), key=lambda v: -v[0])[:3]
        ut.append({
            "id": aid,
            "namn": prof.get("display_name") or info[aid]["namn"],
            "larosate": k["inst"].get("display_name", ""),
            "land": k["land"],
            "poang": poang[aid],
            "h_index": h,
            "antal_verk": verk,
            "forsta_ar": forsta,
            "akad_alder": akad,
            "senioritet": senioritet(akad, h, verk),
            "varning": profilvarning(akad, verk),
            # Temastyrka normaliserad mot den starkaste kandidaten per tema.
            "temastyrka": {t: s / maxtema[t] for t, s in info[aid]["teman"].items() if maxtema[t]},
            "toppverk": [(v, t, a, d) for v, t, a, d in toppverk],
            "amnen": [t.get("display_name", "") for t in (prof.get("topics") or [])[:3]],
            "openalex_url": f"https://openalex.org/{kort_id(aid)}",
            "orcid": prof.get("orcid") or "",
        })
    ut.sort(key=lambda k: -k["poang"])
    for i, k in enumerate(ut, 1):
        k["rang"] = i

    grupp = foresla_grupp(ut, len(teman), args.gruppstorlek, args.max_per_land)
    skriv_excel(args.ut, ut, grupp, teman, args, oa.antal_anrop)

    ej_tackta = [i for i in range(len(teman))
                 if not any(k["temastyrka"].get(i, 0) >= 0.3 for k in grupp)]
    print(f"\nKlart: {len(ut)} kandidater, gruppförslag med {len(grupp)} personer → {args.ut}",
          file=sys.stderr)
    if ej_tackta:
        print("Teman som gruppförslaget inte täcker väl: " +
              "; ".join(kort_etikett(teman[i], 50) for i in ej_tackta), file=sys.stderr)
    print(f"API-anrop denna körning: {oa.antal_anrop} (resten från cache).", file=sys.stderr)


if __name__ == "__main__":
    main()
