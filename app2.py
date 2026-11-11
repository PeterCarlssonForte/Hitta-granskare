"""Streamlit-app för att hitta europeiska granskare till en utlysning.

Starta med (i mappen beredningsverktyg):
    streamlit run rekryteraapp.py

All söklogik ligger i rekrytera.py – appen är bara ett grafiskt skal runt den.
"""

from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rekrytera as r  # noqa: E402
from openalex import EUROPA, OpenAlex, _las_api_nyckel, kort_id  # noqa: E402

st.set_page_config(page_title="Rekrytera granskare", layout="wide")


# ---------------------------------------------------------------- hjälpfunktioner
def tolka_teman(utlysning: str, tematext: str) -> list[str]:
    rader = [x.strip(" -•\t") for x in tematext.splitlines()]
    teman = [x for x in rader if x and not x.startswith("#")]
    return teman or r.las_teman(utlysning, None)


def kor_sokning(utlysning: str, teman: list[str], api_key: str | None, fran_ar: int,
                per_sokning: int, antal_kandidater: int, exkludera: list[str],
                gruppstorlek: int, max_per_land: int) -> dict | None:
    oa = OpenAlex(api_key=api_key or None)

    with st.status("Söker granskare …", expanded=True) as status:
        st.write(f"Söker i OpenAlex med {len(teman)} teman (kan ta en minut) …")
        poang, info = r.samla_kandidater(oa, teman, utlysning, fran_ar, per_sokning)
        if not poang:
            status.update(label="Inga träffar", state="error")
            st.error("Inga träffar. Prova bredare teman eller ett tidigare startår.")
            return None

        topp = sorted(poang, key=lambda a: -poang[a])[: int(antal_kandidater * 1.5)]
        st.write(f"Hämtar profiler för {len(topp)} forskare …")
        profiler = {p["id"]: p for p in oa.hamta_forfattare(topp, r.FORF_SELECT)}

        kandidater = []
        for aid in topp:
            prof = profiler.get(aid)
            if not prof:
                continue
            inst = (prof.get("last_known_institutions") or [{}])[0] or {}
            land = (inst.get("country_code") or "").upper()
            if land not in EUROPA or land in exkludera:
                continue
            kandidater.append({"id": aid, "prof": prof, "inst": inst, "land": land})
            if len(kandidater) >= antal_kandidater:
                break

        maxtema: dict[int, float] = defaultdict(float)
        for k in kandidater:
            for t, s in info[k["id"]]["teman"].items():
                maxtema[t] = max(maxtema[t], s)

        st.write(f"Uppskattar akademisk ålder för {len(kandidater)} kandidater …")
        stapel = st.progress(0.0)
        ut = []
        for i, k in enumerate(kandidater, 1):
            prof, aid = k["prof"], k["id"]
            forsta = oa.forsta_publiceringsar(aid)
            akad = (r.AR - forsta) if forsta else None
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
                "senioritet": r.senioritet(akad, h, verk),
                "varning": r.profilvarning(akad, verk),
                "temastyrka": {t: s / maxtema[t] for t, s in info[aid]["teman"].items()
                               if maxtema[t]},
                "toppverk": toppverk,
                "amnen": [t.get("display_name", "") for t in (prof.get("topics") or [])[:3]],
                "openalex_url": f"https://openalex.org/{kort_id(aid)}",
                "orcid": prof.get("orcid") or "",
            })
            stapel.progress(i / len(kandidater))

        ut.sort(key=lambda k: -k["poang"])
        for i, k in enumerate(ut, 1):
            k["rang"] = i

        grupp = r.foresla_grupp(ut, len(teman), gruppstorlek, max_per_land)

        buf = io.BytesIO()
        args = SimpleNamespace(fran_ar=fran_ar, exkludera_lander=exkludera)
        r.skriv_excel(buf, ut, grupp, teman, args, oa.antal_anrop)

        status.update(label=f"Klart – {len(ut)} kandidater, {oa.antal_anrop} nya API-anrop",
                      state="complete", expanded=False)

    return {"kandidater": ut, "grupp": grupp, "teman": teman, "excel": buf.getvalue()}


def kandidattabell(kandidater: list[dict], teman: list[str]) -> pd.DataFrame:
    kolumner = [namn for namn, _ in r.KAND_KOLUMNER]
    return pd.DataFrame([r.kandidatrad(k, teman) for k in kandidater], columns=kolumner)


def visa_kandidater(df: pd.DataFrame):
    st.dataframe(
        df, hide_index=True,
        column_config={
            "OpenAlex-profil": st.column_config.LinkColumn(display_text="Profil"),
            "ORCID": st.column_config.LinkColumn(display_text="ORCID"),
            "Matchning": st.column_config.NumberColumn(format="%.2f"),
        },
    )


# ---------------------------------------------------------------- sidopanel
with st.sidebar:
    st.header("Inställningar")
    api_key = st.text_input("OpenAlex API-nyckel", value=_las_api_nyckel() or "",
                            type="password",
                            help="Gratis nyckel på openalex.org/settings/api. "
                                 "Utan nyckel räcker dagskvoten bara till ett fåtal anrop.")
    fran_ar = st.number_input("Publikationer från år", min_value=1950, max_value=r.AR,
                              value=r.AR - 10, step=1)
    exkludera_txt = st.text_input("Uteslut länder (landskoder)", value="SE",
                                  help="Kommaseparerat, t.ex. SE,NO")
    gruppstorlek = st.number_input("Gruppstorlek", min_value=1, max_value=50, value=10)
    max_per_land = st.number_input("Max per land i gruppen", min_value=1, max_value=20, value=2)
    with st.expander("Avancerat"):
        antal_kandidater = st.number_input("Antal kandidater att lista", min_value=10,
                                           max_value=500, value=150, step=10)
        per_sokning = st.number_input("Publikationer per sökfråga", min_value=10,
                                      max_value=200, value=50, step=10,
                                      help="Semantisk sökning tar max 50.")


# ---------------------------------------------------------------- huvudvy
st.title("Hitta granskare")
st.caption("Söker europeiska forskare i OpenAlex som passar utlysningen och föreslår "
           "en beredningsgrupp. Docentnivå är en grov uppskattning som måste verifieras.")

vanster, hoger = st.columns(2)
with vanster:
    fil = st.file_uploader("Ladda upp utlysningen (.txt)", type=["txt"])
    standard = fil.getvalue().decode("utf-8", errors="replace") if fil else ""
    utlysning = st.text_area("… eller klistra in utlysningstexten", value=standard, height=300)
with hoger:
    tematext = st.text_area(
        "Teman (valfritt, ett per rad, helst på engelska)", height=300,
        placeholder="social services prevention\nevidence-based social work\n…",
        help="Lämnas tomt delas utlysningen upp i stycken automatiskt.")

if st.button("Sök granskare", type="primary", disabled=not utlysning.strip()):
    teman = tolka_teman(utlysning, tematext)
    if any(r.ser_svensk_ut(t) for t in teman):
        st.info("Tips: OpenAlex innehåll är mest på engelska – teman på engelska ger "
                "oftast bättre träffar.")
    exkludera = [x.strip().upper() for x in exkludera_txt.split(",") if x.strip()]
    st.session_state["resultat"] = kor_sokning(
        utlysning, teman, api_key.strip(), int(fran_ar), int(per_sokning),
        int(antal_kandidater), exkludera, int(gruppstorlek), int(max_per_land))

res = st.session_state.get("resultat")
if res:
    kandidater, grupp, teman = res["kandidater"], res["grupp"], res["teman"]

    st.download_button("Ladda ner Excel", data=res["excel"], file_name="kandidater.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")

    ej_tackta = [i for i in range(len(teman))
                 if not any(k["temastyrka"].get(i, 0) >= 0.3 for k in grupp)]
    if ej_tackta:
        st.warning("Teman som gruppförslaget inte täcker väl: " +
                   "; ".join(r.kort_etikett(teman[i], 60) for i in ej_tackta))

    flik_grupp, flik_alla, flik_tema = st.tabs(
        [f"Förslag grupp ({len(grupp)})", f"Alla kandidater ({len(kandidater)})", "Per tema"])

    with flik_grupp:
        if grupp:
            visa_kandidater(kandidattabell(grupp, teman))
        else:
            st.info("Ingen kandidat bedömdes nå docentnivå (Trolig/Möjlig).")

    with flik_alla:
        df = kandidattabell(kandidater, teman)
        nivaer = st.multiselect("Visa docentnivå", ["Trolig", "Möjlig", "Osäker/yngre"],
                                default=["Trolig", "Möjlig", "Osäker/yngre"])
        visa_kandidater(df[df["Docentnivå?"].isin(nivaer)])

    with flik_tema:
        for t, tema in enumerate(teman):
            basta = sorted((k for k in kandidater if k["temastyrka"].get(t, 0) > 0),
                           key=lambda k: -k["temastyrka"][t])[:10]
            i_grupp = [k["namn"] for k in grupp if k["temastyrka"].get(t, 0) >= 0.3]
            with st.expander(f"{t + 1}. {r.kort_etikett(tema, 100)}"):
                st.markdown("**Täcks i gruppförslaget av:** " +
                            (", ".join(i_grupp) or ":red[ej täckt]"))
                st.markdown("\n".join(f"- {k['namn']} ({k['land']}, {k['senioritet']})"
                                      for k in basta) or "Inga kandidater.")
