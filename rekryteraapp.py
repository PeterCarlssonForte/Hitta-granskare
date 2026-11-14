"""Streamlit-app för att hitta europeiska granskare till en utlysning.

Starta med (i mappen beredningsverktyg):
    streamlit run rekryteraapp.py

All söklogik ligger i rekrytera.py – appen är bara ett grafiskt skal runt den.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rekrytera as r  # noqa: E402
import utvardera as u  # noqa: E402
from openalex import OpenAlex, OpenAlexStopp, _las_api_nyckel  # noqa: E402
from teman import ai_prompt, granska_teman, tolka_teman  # noqa: E402

st.set_page_config(page_title="Rekrytera granskare", layout="wide")

EXEMPEL = (
    "school-based mental health promotion | Universal and targeted interventions delivered in schools "
    "to promote mental wellbeing and prevent mental ill-health among children and adolescents, "
    "including evaluations of their effectiveness and implementation.\n"
    "suicide prevention | …\n"
    "EJ: clinical drug trials | Pharmacological treatment studies and biomedical research on mental disorders.")

DJUPTEXT = {
    "snabb": "Snabb – en sökning per tema, 150 profilkontroller",
    "normal": "Normal – tre tidsperioder per tema, 300 profilkontroller",
    "grundlig": "Grundlig – dubbla sökningar, 500 profilkontroller",
}
LIKHETSTEXT = {
    "auto": "Språkmodell om den finns installerad, annars ordjämförelse",
    "modell": "Språkmodell (laddas ner vid behov)",
    "tfidf": "Ordjämförelse (snabbast)",
}


def dollar(belopp: float) -> str:
    return f"{belopp:.2f}".replace(".", ",")


# ---------------------------------------------------------------- hjälpfunktioner
def kor_sokning(teman, undantag, api_key: str, kostnad: float, **installningar) -> dict | None:
    oa = OpenAlex(api_key=api_key or None)
    with st.status("Söker granskare …", expanded=True) as status:
        rad = st.empty()
        stapel = st.progress(0.0)

        def framsteg(text: str, andel: float | None = None):
            rad.write(text)
            if andel is not None:
                stapel.progress(min(1.0, max(0.0, andel)))

        try:
            kvar = oa.kolla_budget()
            if kvar is not None and kvar < kostnad:
                st.warning(f"Dagens budget hos OpenAlex kan ta slut under körningen (kvar: {dollar(kvar)} "
                           f"dollar, körningen kostar upp till {dollar(kostnad)}). Det som hinner hämtas "
                           "sparas, så en ny körning i morgon fortsätter där den här slutade.")
            res = r.kor(oa, teman, undantag, framsteg=framsteg, **installningar)
        except r.IngaTraffar as fel:
            status.update(label="Inga träffar", state="error")
            st.error(str(fel))
            return None
        except OpenAlexStopp as fel:
            status.update(label="Avbrutet", state="error")
            st.error(str(fel))
            return None

        buf = io.BytesIO()
        r.skriv_excel(buf, res)
        res["excel"] = buf.getvalue()
        stapel.progress(1.0)
        status.update(label=f"Klart – {len(res['kandidater'])} kandidater, {oa.antal_anrop} nya API-anrop "
                            f"({dollar(oa.kostnad_usd)} dollar)", state="complete", expanded=False)
    return res


def visa_tabell(rader: list[list], kolumner: list[tuple[str, int]]):
    df = pd.DataFrame(rader, columns=[namn for namn, _ in kolumner])
    st.dataframe(
        df, hide_index=True,
        column_config={
            "OpenAlex-profil": st.column_config.LinkColumn(display_text="Profil"),
            "ORCID": st.column_config.LinkColumn(display_text="ORCID"),
            "Matchning": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def las_in_temafil():
    fil = st.session_state.get("temafil")
    if fil is not None:
        st.session_state["tematext"] = fil.getvalue().decode("utf-8-sig", errors="replace")


# ---------------------------------------------------------------- sidopanel
with st.sidebar:
    st.header("Inställningar")
    api_key = st.text_input("OpenAlex API-nyckel", value=_las_api_nyckel() or "",
                            type="password",
                            help="Gratis nyckel på openalex.org/settings/api. Den ger en egen "
                                 "dagsbudget på 1 dollar, vilket räcker till ungefär tio körningar.")
    if not api_key.strip():
        st.warning("Utan nyckel är dagsbudgeten 0,10 dollar och delas med andra på nätverket. "
                   "Det räcker till ungefär en snabb körning.")
    # Utan nyckel räcker dagsbudgeten bara till en snabb körning – börja där.
    djup = st.selectbox("Sökdjup", list(r.DJUP), index=1 if api_key.strip() else 0,
                        format_func=DJUPTEXT.get,
                        help="Djupare sökning hittar fler kandidater men kostar mer av dagsbudgeten.")
    fran_ar = st.number_input("Publikationer från år", min_value=1950, max_value=r.AR,
                              value=r.AR - 10, step=1)
    exkludera_txt = st.text_input("Uteslut länder (landskoder)", value="SE",
                                  help="Kommaseparerat, t.ex. SE,NO. Andelen verk med medförfattare "
                                       "i dessa länder visas också som tidig jävsindikator.")
    gruppstorlek = st.number_input("Gruppstorlek", min_value=1, max_value=50, value=10)
    max_per_land = st.number_input("Max per land i gruppen", min_value=1, max_value=20, value=2)
    with st.expander("Avancerat"):
        antal_kandidater = st.number_input("Antal kandidater att lista", min_value=10,
                                           max_value=500, value=150, step=10)
        likhet = st.selectbox("Profilkontrollens jämförelse", list(LIKHETSTEXT),
                              format_func=LIKHETSTEXT.get,
                              help="Språkmodellen förstår synonymer och är träffsäkrare. Första körningen "
                                   "tar några minuter; därefter sparas resultatet.")


# ---------------------------------------------------------------- huvudvy
st.title("Hitta granskare")
st.caption("Söker europeiska forskare i OpenAlex och bedömer deras egna publikationer mot de kompetenser "
           "beredningsgruppen behöver. Docentnivå är en grov uppskattning som måste verifieras.")

st.subheader("1. Utlysningen")
fil = st.file_uploader("Ladda upp utlysningen (.txt)", type=["txt"])
standard = fil.getvalue().decode("utf-8-sig", errors="replace") if fil else ""
utlysning = st.text_area("… eller klistra in utlysningstexten", value=standard, height=180,
                         help="Används bara för att skapa en fråga till en AI-assistent. "
                              "Sökningen utgår från temana.")

st.subheader("2. Teman – de kompetenser gruppen behöver")
st.markdown("En rad per tema: `kort sökfras | beskrivning`. Sökfrasen (2–6 ord) används för nyckelordssökning. "
            "Beskrivningen (2–3 meningar, skriven som ett abstract) används för semantisk sökning och när "
            "varje forskares publikationer bedöms. Rader som börjar med `EJ:` är forskning som utlysningen "
            "inte finansierar. Skriv på engelska.")
with st.expander("Få ett temaförslag från en AI-assistent"):
    if utlysning.strip():
        st.markdown("1. Kopiera frågan nedan med knappen uppe till höger i rutan.\n"
                    "2. Klistra in den i den AI-assistent ni får använda.\n"
                    "3. Klistra in svaret i temarutan och justera det. Ni kan området bäst.")
        st.code(ai_prompt(utlysning), language=None, wrap_lines=True)
    else:
        st.info("Lägg in utlysningstexten ovan, så skapas en färdig fråga här.")
st.file_uploader("Ladda upp en temafil (.txt)", type=["txt"], key="temafil", on_change=las_in_temafil)
tematext = st.text_area("Teman", key="tematext", height=260, placeholder=EXEMPEL)
teman, undantag = tolka_teman(tematext)

if teman or undantag:
    st.dataframe(pd.DataFrame(
        [{"Typ": "Tema", "Sökfras": t.fras, "Beskrivning": t.beskrivning} for t in teman]
        + [{"Typ": "Undantag", "Sökfras": t.fras, "Beskrivning": t.beskrivning} for t in undantag]),
        hide_index=True)
if tematext.strip() or undantag:
    for varning in granska_teman(teman, undantag):
        st.warning(varning)

st.subheader("3. Sök")
exkludera = [x.strip().upper() for x in exkludera_txt.split(",") if x.strip()]
kostnad = r.uppskatta_kostnad(len(teman), len(undantag), djup, int(antal_kandidater))
if teman:
    st.caption(f"Körningen kostar högst cirka {dollar(kostnad)} dollar av OpenAlex dagsbudget "
               f"({'1 dollar med nyckel' if api_key.strip() else '0,10 dollar utan nyckel'}). "
               "Svar som redan finns i cachen kostar inget.")
else:
    st.info("Skriv minst ett tema för att kunna söka.")

if st.button("Sök granskare", type="primary", disabled=not teman):
    st.session_state["resultat"] = kor_sokning(
        teman, undantag, api_key.strip(), kostnad,
        fran_ar=int(fran_ar), djup=djup, antal_kandidater=int(antal_kandidater), exkludera=exkludera,
        gruppstorlek=int(gruppstorlek), max_per_land=int(max_per_land), likhet=likhet)

res = st.session_state.get("resultat")
if res:
    kandidater, grupp, teman_res, m = res["kandidater"], res["grupp"], res["teman"], res["metod"]

    st.download_button("Ladda ner Excel", data=res["excel"], file_name="kandidater.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")
    st.caption(f"Sökningen gav {m['antal_steg1']} europeiska författare. {m['antal_profilerade']} "
               f"profilkontrollerades och {m['antal_bedomda']} bedömdes. {m['antal_valbara']} av de listade "
               f"kan föreslås till gruppen. Profilkontroll: {m['likhet']}.")

    if len(grupp) < m["gruppstorlek"]:
        st.warning(f"Bara {len(grupp)} av {m['gruppstorlek']} platser kunde fyllas. Prova ett djupare "
                   "sökdjup, fler per land eller bredare teman.")
    if res["tunt"]:
        st.warning("Teman som färre än två i gruppförslaget täcker: " + "; ".join(
            f"{r.kort_etikett(teman_res[t].etikett, 60)} ({n})" for t, n in res["tunt"]))
    if m["inaktiva_undantag"]:
        st.info("Dessa undantag kunde inte användas (för få sökträffar): " + "; ".join(m["inaktiva_undantag"]))

    flik_grupp, flik_alla, flik_tema, flik_facit = st.tabs(
        [f"Förslag grupp ({len(grupp)})", f"Alla kandidater ({len(kandidater)})", "Per tema",
         "Mät mot facit"])

    with flik_grupp:
        if grupp:
            visa_tabell([r.grupprad(k, teman_res, res["reserver"].get(k["id"], [])) for k in grupp],
                        r.GRUPP_KOLUMNER)
            st.caption("Reserverna har en temaprofil som liknar ledamotens och har inte publicerat "
                       "tillsammans med någon av de övriga i gruppen.")
        else:
            st.info(f"Ingen kandidat uppfyller kraven för gruppförslaget: minst {r.MIN_RELEVANTA} "
                    "relevanta verk och docentnivå Trolig eller Möjlig.")

    with flik_alla:
        vanster, hoger = st.columns([3, 1])
        nivaer = vanster.multiselect("Visa docentnivå", ["Trolig", "Möjlig", "Osäker/yngre"],
                                     default=["Trolig", "Möjlig", "Osäker/yngre"])
        bara_valbara = hoger.checkbox("Bara valbara till gruppen")
        visa_tabell([r.kandidatrad(k, teman_res) for k in kandidater
                     if k["senioritet"] in nivaer and (k["valbar"] or not bara_valbara)],
                    r.KAND_KOLUMNER)

    with flik_tema:
        st.dataframe(pd.DataFrame(r.tematackning(res), columns=[
            "Tema", "Täcks i gruppförslaget av (antal relevanta verk)", "Starkaste kandidaterna"]),
            hide_index=True)

    with flik_facit:
        st.markdown("Kör verktyget på en avslutad utlysning och klistra in dem som satt i "
                    "beredningsgruppen, en per rad med ORCID eller OpenAlex-id (gärna med namn). "
                    "Då syns var de hamnar, och om en ändring av teman eller inställningar hjälper.")
        facittext = st.text_area("Facit", height=140, key="facittext",
                                 placeholder="Anna Andersson 0000-0002-1825-0097\nA5023888391")
        if st.button("Mät", disabled=not facittext.strip()):
            rader, s = u.utvardera(res, u.tolka_facit(facittext), OpenAlex(api_key=api_key.strip() or None))
            st.markdown("**" + u.sammanfatta(s) + "**")
            st.dataframe(pd.DataFrame(rader), hide_index=True)
