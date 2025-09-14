import streamlit as st
import requests
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Forskar-matchare", layout="wide")
st.title("🔍 Forskar-matchare (Concepts + Keywords)")

# ---------------- INPUT ----------------
call_text = st.text_area("Klistra in utlysningstext här:", height=200)

num_per_source = st.number_input(
    "Antal forskare att hämta per källa (per keyword/concept):",
    min_value=5, max_value=200, value=20, step=5
)
top_n = st.number_input(
    "Visa topp N forskare i slutlistan:",
    min_value=5, max_value=200, value=50, step=5
)
exclude_countries = st.multiselect(
    "Exkludera forskare från följande länder (ISO landkod):",
    ["US", "CN", "DE", "GB", "SE", "FR", "NL", "IT", "NO", "FI"]
)

# ---------- Rankingmetod dropdown UTANFÖR knappen ----------
rank_method = st.selectbox(
    "Sortera forskare efter:",
    ["Citeringar", "Publikationer", "Kombinerat", "Snabb matchning", "AI-semantisk matchning"]
)

# ---------- Funktioner ----------
def find_concepts(keyword):
    url = f"https://api.openalex.org/concepts?search={keyword}"
    r = requests.get(url)
    return r.json().get("results", [])

def find_authors_by_concept(concept_id, per_page=20):
    url = f"https://api.openalex.org/authors?filter=concepts.id:{concept_id}&per_page={per_page}"
    r = requests.get(url)
    return r.json().get("results", [])

def find_authors_by_keyword(keyword, per_page=20):
    url = f"https://api.openalex.org/works?search={keyword}&per_page={per_page}"
    r = requests.get(url).json()
    authors = []
    for work in r.get("results", []):
        for a in work.get("authorships", []):
            author = a.get("author", {})
            if author:
                institutions = a.get("institutions", [])
                inst_name = institutions[0].get("display_name", "Okänd institution") if institutions else "Okänd institution"
                country = institutions[0].get("country_code", "Okänt land") if institutions else "Okänt land"
                authors.append({
                    "name": author.get("display_name", "Okänd"),
                    "Profil": author.get("id", ""),
                    "institution": inst_name,
                    "Land": country,
                    "works": None,
                    "citations": None,
                    "match_source": "keyword",
                    "matched_keyword": keyword
                })
    return authors

# ---------- Hämta forskare när knappen trycks ----------
if st.button("Hitta forskare") and call_text.strip():
    with st.spinner("Bearbetar texten..."):
        # 1. Extrahera nyckelord
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            call_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5
        )
        keywords_list = [kw for kw, _ in keywords]
        st.subheader("🎯 Identifierade nyckelord")
        st.write(keywords_list)

        # 2. Hämta forskare från concepts
        concept_authors = []
        for kw in keywords_list:
            results = find_concepts(kw)
            if results:
                concept_id = results[0]["id"].split("/")[-1]
                concept_authors.extend(find_authors_by_concept(concept_id, per_page=num_per_source))

        # 3. Hämta forskare från keywords
        keyword_authors = []
        for kw in keywords_list:
            keyword_authors.extend(find_authors_by_keyword(kw, per_page=num_per_source))

        # 4. Kombinera
        combined = {}
        for a in concept_authors:
            name = a.get("display_name", "Okänd")
            url = a.get("id", "")
            institutions = a.get("last_known_institutions", [])
            inst_name = institutions[0].get("display_name", "Okänd institution") if institutions else "Okänd institution"
            country = institutions[0].get("country_code", "Okänt land") if institutions else "Okänt land"
            works = a.get("works_count", 0)
            citations = a.get("cited_by_count", 0)
            if name in combined:
                combined[name]["match_source"].add("concept")
                combined[name]["matched_terms"].append(kw)
            else:
                combined[name] = {
                    "Namn": name,
                    "Profil": url,
                    "Institution": inst_name,
                    "Land": country,
                    "Publikationer": works,
                    "Citeringar": citations,
                    "match_source": {"concept"},
                    "matched_terms": [kw]
                }

        for a in keyword_authors:
            name = a["name"]
            keyword_term = a.get("matched_keyword", "keyword")
            if name in combined:
                combined[name]["match_source"].add("keyword")
                combined[name]["matched_terms"].append(keyword_term)
            else:
                combined[name] = {
                    "Namn": a["name"],
                    "Profil": a["Profil"],
                    "Institution": a["institution"],
                    "Land": a["Land"],
                    "Publikationer": None,
                    "Citeringar": None,
                    "match_source": {"keyword"},
                    "matched_terms": [keyword_term]
                }

        # 5. DataFrame
        df = pd.DataFrame(combined.values())
        df["match_source"] = df["match_source"].apply(lambda x: ", ".join(x))
        df["matched_terms"] = df["matched_terms"].apply(lambda x: ", ".join(x))

        # Exkludera länder
        if exclude_countries:
            df = df[~df["Land"].isin(exclude_countries)]

        # Spara i session_state
        st.session_state['df_forskare'] = df

# ---------- 6. Visa och sortera redan hämtad data ----------
if 'df_forskare' in st.session_state:
    df_to_show = st.session_state['df_forskare']

    if rank_method == "Citeringar":
        df_to_show = df_to_show.sort_values("Citeringar", ascending=False, na_position="last")
    elif rank_method == "Publikationer":
        df_to_show = df_to_show.sort_values("Publikationer", ascending=False, na_position="last")
    elif rank_method == "Kombinerat":
        df_to_show["score"] = (
            (df_to_show["Citeringar"].fillna(0) / df_to_show["Citeringar"].fillna(0).max()) +
            (df_to_show["Publikationer"].fillna(0) / df_to_show["Publikationer"].fillna(0).max())
        )
        df_to_show = df_to_show.sort_values("score", ascending=False, na_position="last")
    elif rank_method == "Snabb matchning":
        df_to_show["match_score"] = df_to_show["matched_terms"].apply(lambda x: len(x.split(",")))
        df_to_show = df_to_show.sort_values(["match_score", "Citeringar"], ascending=[False, False])
    else:  # AI-semantisk matchning
        model = st.cache_resource(lambda: SentenceTransformer("all-MiniLM-L6-v2"))()
        call_embedding = model.encode(call_text, convert_to_tensor=True)
        df_to_show["profile_text"] = df_to_show["Namn"].fillna("") + " " + df_to_show["Institution"].fillna("") + " " + df_to_show["matched_terms"].fillna("")
        df_to_show["similarity"] = df_to_show["profile_text"].apply(
            lambda x: util.cos_sim(call_embedding, model.encode(x, convert_to_tensor=True)).item()
        )
        df_to_show = df_to_show.sort_values("similarity", ascending=False, na_position="last")

    df_to_show = df_to_show.head(top_n)
    st.subheader("🏆 Resultatlista")
    st.dataframe(df_to_show)

    # Export
    st.download_button(
        label="📥 Ladda ner som CSV",
        data=df_to_show.to_csv(index=False).encode("utf-8"),
        file_name="forskarlista.csv",
        mime="text/csv"
    )
    st.download_button(
        label="📥 Ladda ner som Excel",
        data=df_to_show.to_excel(index=False, engine="openpyxl"),
        file_name="forskarlista.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



