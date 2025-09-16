import streamlit as st
import requests
import pandas as pd
from keybert import KeyBERT
import datetime
from sentence_transformers import SentenceTransformer, util

# Initiera modeller
kw_model = KeyBERT()
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Laddas på CPU

st.title("🔎 Hitta forskare för utlysning")

# ---------- Inputs ----------
call_text = st.text_area("Klistra in utlysningstext här:")

num_keywords = st.slider("Hur många sökord vill du extrahera?", 5, 30, 10)
ngram_range = st.slider("Max längd på fraser (antal ord)", 1, 4, 2)
num_per_source = st.slider("Hur många forskare per sökord/ämne?", 5, 50, 20)

include_countries = st.multiselect(
    "Inkludera endast dessa länder (frivilligt, filtreras efter hämtning):",
    options=["SE", "NO", "DK", "FI", "US", "DE", "UK", "FR", "NL", "IT"]
)

exclude_countries = st.multiselect(
    "Exkludera forskare från dessa länder:",
    options=["SE", "NO", "DK", "FI", "US", "DE", "UK", "FR", "NL", "IT"]
)

ranking_method = st.selectbox(
    "Sortera forskare efter:",
    ["Citeringar", "Publikationer", "Semantisk AI-matchning"]
)

# ---------- Hjälpfunktioner ----------
def fetch_author_details(author_id):
    r = requests.get(f"https://api.openalex.org/authors/{author_id.split('/')[-1]}")
    if r.status_code == 200:
        return r.json()
    return {}

def is_active_recently(author_json, years=5):
    current_year = datetime.datetime.now().year
    counts = author_json.get("counts_by_year", [])
    recent = [c for c in counts if c["year"] >= current_year - years + 1]
    total_recent_pubs = sum(c["works_count"] for c in recent)
    return total_recent_pubs > 0

def find_authors_by_concept(concept_id, per_page=20):
    url = f"https://api.openalex.org/authors?filter=concepts.id:{concept_id}&per-page={per_page}"
    r = requests.get(url)
    return r.json().get("results", [])

def find_authors_by_keyword(keyword, per_page=20):
    url = f"https://api.openalex.org/works?search={keyword}&per-page={per_page}"
    r = requests.get(url).json()
    authors = []
    for work in r.get("results", []):
        for a in work.get("authorships", []):
            author = a.get("author", {})
            if author:
                authors.append({
                    "id": author.get("id", ""),
                    "name": author.get("display_name", "Okänd"),
                    "matched_term": keyword
                })
    return authors

# ---------- Steg 1: Extrahera och välj sökord ----------
if st.button("🔑 Extrahera sökord") or 'extracted_keywords' in st.session_state:
    if call_text.strip():
        if 'extracted_keywords' not in st.session_state:
            keywords = kw_model.extract_keywords(
                call_text,
                keyphrase_ngram_range=(1, ngram_range),
                stop_words="english",
                top_n=num_keywords
            )
            st.session_state['extracted_keywords'] = [kw[0] for kw in keywords]
            st.session_state['selected_keywords'] = st.session_state['extracted_keywords'].copy()

        # Visa checkboxar
        st.write("### Identifierade sökord/fraser (välj de som är relevanta)")
        for kw in st.session_state['extracted_keywords']:
            checked = kw in st.session_state['selected_keywords']
            val = st.checkbox(kw, value=checked, key=f"chk_{kw}")
            if val and kw not in st.session_state['selected_keywords']:
                st.session_state['selected_keywords'].append(kw)
            elif not val and kw in st.session_state['selected_keywords']:
                st.session_state['selected_keywords'].remove(kw)
    else:
        st.warning("Klistra in utlysningstext först.")

# ---------- Steg 2: Hämta forskare ----------
if st.button("✅ Hämta forskare"):
    selected_keywords = st.session_state.get('selected_keywords', [])
    if not selected_keywords:
        st.warning("Välj minst ett sökord från steg 1.")
    else:
        with st.spinner("Hämtar forskare..."):
            # Hitta concepts
            concepts = []
            for kw in selected_keywords:
                r = requests.get(f"https://api.openalex.org/concepts?search={kw}")
                results = r.json().get("results", [])
                if results:
                    concepts.append(results[0])

            # Hämta forskare
            temp_authors = {}

            # Concepts
            for c in concepts:
                for a in find_authors_by_concept(c["id"], per_page=num_per_source):
                    author_id = a.get("id", "")
                    if not author_id:
                        continue
                    if author_id not in temp_authors:
                        temp_authors[author_id] = {"matched_terms": set(), "author_obj": a}
                    temp_authors[author_id]["matched_terms"].add(c.get("display_name", ""))

            # Keywords
            for kw in selected_keywords:
                for a in find_authors_by_keyword(kw, per_page=num_per_source):
                    author_id = a.get("id", "")
                    if not author_id:
                        continue
                    if author_id not in temp_authors:
                        temp_authors[author_id] = {"matched_terms": set(), "author_obj": a}
                    temp_authors[author_id]["matched_terms"].add(kw)

            # Skapa lista med fulla profiler
            author_list = []
            for author_id, info in temp_authors.items():
                details = fetch_author_details(author_id)
                if not details or not is_active_recently(details, years=5):
                    continue

                institutions = details.get("last_known_institutions", [])
                inst_name = institutions[0].get("display_name", "Okänd institution") if institutions else "Okänd institution"
                country = institutions[0].get("country_code", "Okänt land") if institutions else "Okänt land"

                works = details.get("works_count", 0)
                citations = details.get("cited_by_count", 0)

                author_list.append({
                    "Namn": details.get("display_name", "Okänd"),
                    "Institution": inst_name,
                    "Land": country,
                    "Publikationer": works,
                    "Citeringar": citations,
                    "Profil": author_id,
                    "Matched_terms": ", ".join(info["matched_terms"])
                })

            df = pd.DataFrame(author_list)

            if df.empty:
                st.warning("Inga forskare hittades med dessa kriterier.")
            else:
                # Landfilter efter hämtning
                if include_countries:
                    df = df[df["Land"].isin(include_countries)]
                if exclude_countries:
                    df = df[~df["Land"].isin(exclude_countries)]

                # Sortering
                if ranking_method == "Citeringar":
                    df = df.sort_values("Citeringar", ascending=False)
                elif ranking_method == "Publikationer":
                    df = df.sort_values("Publikationer", ascending=False)
                elif ranking_method == "Semantisk AI-matchning":
                    # Encode på CPU för att undvika meta tensor-fel
                    query_vec = embed_model.encode(call_text, convert_to_tensor=True, device="cpu")
                    author_texts = (df["Namn"] + " " + df["Institution"] + " " + df["Matched_terms"]).tolist()
                    author_vecs = embed_model.encode(author_texts, convert_to_tensor=True, device="cpu")
                    sims = util.cos_sim(query_vec, author_vecs)[0].cpu().tolist()
                    df["Relevans"] = sims
                    df = df.sort_values("Relevans", ascending=False)

                # Visa tabell med klickbara länkar
                st.data_editor(
                    df,
                    column_config={
                        "Profil": st.column_config.LinkColumn("Profil", display_text="Öppna profil")
                    },
                    hide_index=True,
                    use_container_width=True
                )




