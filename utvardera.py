"""Mät träffsäkerheten mot ett facit.

Kör verktyget på en avslutad utlysning där ni vet vilka som satt i
beredningsgruppen (eller vilka som var bra granskare) och se var de hamnar.
Gör om mätningen när ni ändrar teman, trösklar eller vikter, så syns det om
ändringen hjälpte.

Exempel:
    python utvardera.py --teman exempel/teman.txt --facit exempel/facit.csv --exkludera-lander SE

Facit är en CSV-fil med en rad per person och kolumnerna namn, orcid och/eller
openalex (OpenAlex-id eller profillänk). ORCID eller OpenAlex-id behövs för en
säker matchning; bara namn ger en osäker namnmatchning.

Rapporten visar för varje person hur långt hen kom:
  listad            – med i kandidatlistan, med rang
  utanför listan    – profilkontrollerad men under listans gräns
  bortfiltrerad     – föll bort på land eller saknade data
  ej profilerad     – hittades av sökningen men kom inte med till profilkontrollen
  inte hittad       – sökningen hittade inte personen (teman eller sökdjup räcker inte)
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import requests

from openalex import OpenAlex

_ORCID = re.compile(r"\b(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])\b")
_OPENALEX = re.compile(r"\b(A\d{6,})\b")


def tolka_facit(text: str) -> list[dict]:
    """Fritext, en person per rad: ORCID och/eller OpenAlex-id, gärna med namn."""
    rader = []
    for rad in (text or "").splitlines():
        rad = rad.strip()
        if not rad or rad.startswith("#"):
            continue
        orcid = _ORCID.search(rad)
        oid = _OPENALEX.search(rad)
        namn = rad
        for m in (orcid, oid):
            if m:
                namn = namn.replace(m.group(0), "")
        namn = re.sub(r"https?://\S*|[;,|\t]", " ", namn)
        rader.append({"namn": " ".join(namn.split()),
                      "orcid": orcid.group(1).upper() if orcid else "",
                      "openalex": oid.group(1) if oid else ""})
    return rader


def las_facit(fil: Path) -> list[dict]:
    text = "\n".join(rad for rad in Path(fil).read_text(encoding="utf-8-sig").splitlines()
                     if rad.strip() and not rad.lstrip().startswith("#"))
    forsta = text.splitlines()[0].lower() if text.strip() else ""
    if not any(k in forsta for k in ("orcid", "openalex", "namn")):
        return tolka_facit(text)
    rader = []
    try:
        dialekt = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
    except csv.Error:       # en enda kolumn: läs resten som fritext
        return tolka_facit("\n".join(text.splitlines()[1:]))
    for r in csv.DictReader(text.splitlines(), dialect=dialekt):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        if not any(r.values()):
            continue
        orcid = _ORCID.search(r.get("orcid", ""))
        oid = _OPENALEX.search(r.get("openalex", ""))
        rader.append({"namn": r.get("namn", ""),
                      "orcid": orcid.group(1).upper() if orcid else "",
                      "openalex": oid.group(1) if oid else ""})
    return rader


def utvardera(res: dict, facit: list[dict], oa: OpenAlex | None = None) -> tuple[list[dict], dict]:
    """Jämför ett resultat från rekrytera.kor() med facit.

    Ger (rader, sammanfattning). Med `oa` slås ORCID upp i OpenAlex (gratis),
    vilket behövs för att se om en person som inte profilkontrollerades ändå
    hittades av sökningen.
    """
    tratt = res["tratt"]
    listade = {k["id"]: k for k in res["kandidater"]}
    grupp = {k["id"] for k in res["grupp"]}
    reserv = {r["id"] for lista in res["reserver"].values() for r in lista}
    namn_till_id: dict[str, list[str]] = {}
    for aid, namn in tratt["namn"].items():
        namn_till_id.setdefault(namn.lower(), []).append(aid)

    rader = []
    for person in facit:
        ids: list[str] = []
        osaker = False
        if person["openalex"]:
            ids.append(person["openalex"])
        if person["orcid"]:
            if person["orcid"] in tratt["orcid"]:
                ids.append(tratt["orcid"][person["orcid"]])
            elif oa is not None:
                try:
                    traff = oa.forfattare_via_orcid(person["orcid"])
                except requests.RequestException:
                    traff = None
                if traff:
                    ids.append(traff["id"].rsplit("/", 1)[-1])
        if not ids and person["namn"]:
            ids = namn_till_id.get(person["namn"].lower(), [])
            osaker = bool(ids)
        ids = list(dict.fromkeys(ids))

        # Välj den profil som kom längst (en person kan ha flera profiler i OpenAlex).
        def steg(aid: str) -> tuple:
            if aid in listade:
                return (0, listade[aid]["rang"])
            if aid in tratt["profilerade"]:
                return (1, tratt["profilerade"][aid])
            if aid in tratt["bortfall"]:
                return (2, tratt["steg1"].get(aid, 0))
            if aid in tratt["steg1"]:
                return (3, tratt["steg1"][aid])
            return (4, 0)

        bast = min(ids, key=steg) if ids else None
        niva, varde = steg(bast) if bast else (4, 0)
        rad = {"Namn": person["namn"] or tratt["namn"].get(bast or "", ""), "ORCID": person["orcid"],
               "OpenAlex": bast or "", "Rang": None, "Matchning": None, "Relevanta verk": None,
               "Docentnivå?": "", "I gruppen": ""}
        if niva == 0:
            k = listade[bast]
            rad.update({"Status": "listad", "Rang": k["rang"], "Matchning": round(k["poang"], 1),
                        "Relevanta verk": k["relevanta"], "Docentnivå?": k["senioritet"],
                        "I gruppen": "ja" if bast in grupp else "reserv" if bast in reserv else
                        "" if k["valbar"] else "ej valbar"})
        elif niva == 1:
            rad.update({"Status": "utanför listan", "Rang": varde})
        elif niva == 2:
            rad["Status"] = f"bortfiltrerad: {tratt['bortfall'][bast]}"
        elif niva == 3:
            rad["Status"] = f"ej profilerad (plats {varde} av {len(tratt['steg1'])} i sökningen)"
        elif ids or person["orcid"] or person["openalex"]:
            rad["Status"] = "inte hittad av sökningen"
        else:
            rad["Status"] = "inte hittad (bara namn angivet – lägg till ORCID för säker matchning)"
        if osaker:
            rad["Status"] += " – namnmatchning, osäker"
        rader.append(rad)

    rang = [r["Rang"] for r in rader if r["Status"].startswith("listad")]
    n = len(rader)
    sammanfattning = {
        "antal": n,
        "listade": len(rang),
        "topp10": sum(1 for x in rang if x <= 10),
        "topp25": sum(1 for x in rang if x <= 25),
        "topp50": sum(1 for x in rang if x <= 50),
        "medianrang": statistics.median(rang) if rang else None,
        "i_gruppen": sum(1 for r in rader if r["I gruppen"] == "ja"),
        "inte_hittade": sum(1 for r in rader if r["Status"].startswith("inte hittad")),
    }
    return rader, sammanfattning


def sammanfatta(s: dict) -> str:
    if not s["antal"]:
        return "Facit är tomt."
    delar = [f"{s['listade']} av {s['antal']} i facit finns i kandidatlistan",
             f"{s['topp50']} bland de 50 främsta, {s['topp25']} bland de 25 främsta och {s['topp10']} bland de 10 främsta"]
    if s["medianrang"] is not None:
        delar.append(f"medianrang {s['medianrang']:g}".replace(".", ","))
    delar.append(f"{s['i_gruppen']} i gruppförslaget")
    if s["inte_hittade"]:
        delar.append(f"{s['inte_hittade']} hittades inte alls av sökningen")
    return "; ".join(delar) + "."


def main(argv=None):
    import rekrytera as r
    from teman import las_temafil

    p = argparse.ArgumentParser(description="Mät hur väl rekrytera.py hittar kända granskare.")
    p.add_argument("--teman", type=Path, required=True, help="Temafil för den avslutade utlysningen")
    p.add_argument("--facit", type=Path, required=True, help="CSV med kolumnerna namn, orcid, openalex")
    p.add_argument("--fran-ar", type=int, default=r.AR - 10)
    p.add_argument("--djup", choices=list(r.DJUP), default="normal")
    p.add_argument("--antal-kandidater", type=int, default=150)
    p.add_argument("--exkludera-lander", default="")
    p.add_argument("--gruppstorlek", type=int, default=10)
    p.add_argument("--max-per-land", type=int, default=2)
    p.add_argument("--likhet", choices=["auto", "modell", "tfidf"], default="auto")
    p.add_argument("--ut", type=Path, help="Spara rapporten som CSV (valfritt)")
    args = p.parse_args(argv)

    teman, undantag = las_temafil(args.teman)
    if not teman:
        raise SystemExit(f"Hittade inga teman i {args.teman}.")
    facit = las_facit(args.facit)
    if not facit:
        raise SystemExit(f"Hittade inga personer i {args.facit}.")

    oa = OpenAlex()
    try:
        res = r.kor(oa, teman, undantag, fran_ar=args.fran_ar, djup=args.djup,
                    antal_kandidater=args.antal_kandidater,
                    exkludera=[l.strip().upper() for l in args.exkludera_lander.split(",") if l.strip()],
                    gruppstorlek=args.gruppstorlek, max_per_land=args.max_per_land, likhet=args.likhet,
                    framsteg=lambda text, andel=None: print(f"  {text}", file=sys.stderr))
    except r.IngaTraffar as fel:
        raise SystemExit(str(fel))
    rader, s = utvardera(res, facit, oa)

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"\n{'Namn':30s} {'Rang':>5s} {'Match':>6s} {'Verk':>5s}  Status")
    for rad in sorted(rader, key=lambda x: (x["Rang"] is None, x["Rang"] or 0)):
        print(f"{rad['Namn'][:30]:30s} {str(rad['Rang'] or ''):>5s} {str(rad['Matchning'] or ''):>6s} "
              f"{str(rad['Relevanta verk'] if rad['Relevanta verk'] is not None else ''):>5s}  {rad['Status']}"
              + (f" – {rad['I gruppen']}" if rad["I gruppen"] else ""))
    print("\n" + sammanfatta(s))
    if args.ut:
        with args.ut.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rader[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(rader)
        print(f"Rapporten sparad → {args.ut}")


if __name__ == "__main__":
    main()
