import streamlit as st
import requests
import pandas as pd
import datetime
import time
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import numpy as np
from typing import List, Dict, Tuple

# ---------- Initiera modeller på CPU ----------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    kw_model = KeyBERT(model=embed_model)
    return embed_model, kw_model

embed_model, kw_model = load_models()

st.title("🔎 Hitta forskare för utlysning (Förbättrad)")
st.caption("Nu med semantisk matchning baserad på forskarnas publikationer!")


# ---------- Inputs ----------
call_text = st.text_area("Klistra in utlysningstext här:")

# Nytt: Egna sökord
custom_keywords_str = st.text_input(
    "Lägg till egna sökord (kommaseparerade, t.ex. arbetsmiljö, stresshantering, digitalisering)",
    value=""
)
custom_keywords = [kw.strip() for kw in custom_keywords_str.split(",") if kw.strip()]

num_keywords = st.slider("Hur många sökord vill du extrahera?", 5, 30, 10)
ngram_range = st.slider("Max längd på fraser (antal ord)", 1, 4, 2)
num_per_source = st.slider("Hur många forskare per sökord/ämne?", 5, 50, 20)

# Nya viktningsinställningar
st.sidebar.header("🎯 Matchningsviktning")
semantic_weight = st.sidebar.slider("Semantisk likhet (abstract-matchning)", 0.0, 1.0, 0.5, 0.1)
keyword_weight = st.sidebar.slider("Antal matchade sökord", 0.0, 1.0, 0.3, 0.1)
impact_weight = st.sidebar.slider("Forskningsimpakt (citeringar)", 0.0, 1.0, 0.2, 0.1)

# Normalisera vikter så de summerar till 1.0
total_weight = semantic_weight + keyword_weight + impact_weight
if total_weight > 0:
    semantic_weight /= total_weight
    keyword_weight /= total_weight
    impact_weight /= total_weight

include_countries = st.multiselect(
    "Inkludera endast dessa länder (frivilligt, filtreras efter hämtning):",
    options=["SE", "NO", "DK", "FI", "US", "DE", "UK", "FR", "NL", "IT"]
)

exclude_countries = st.multiselect(
    "Exkludera forskare från dessa länder:",
    options=["SE", "NO", "DK", "FI", "US", "DE", "UK", "FR", "NL", "IT"]
)

# ---------- Förbättrade hjälpfunktioner ----------
@st.cache_data(ttl=3600)  # Cache i 1 timme
def fetch_author_details(author_id: str) -> Dict:
    """Hämta detaljerad information om en författare"""
    try:
        clean_id = author_id.split('/')[-1] if 'openalex.org' in author_id else author_id
        response = requests.get(f"https://api.openalex.org/authors/{clean_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.warning(f"Kunde inte hämta data för författare {author_id}: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_author_research_profile(author_id: str, num_works: int = 5) -> str:
    """Hämta författarens forsknungsprofil baserad på senaste publikationer"""
    try:
        clean_id = author_id.split('/')[-1] if 'openalex.org' in author_id else author_id
        works_url = f"https://api.openalex.org/works?filter=author.id:{clean_id}&sort=publication_year:desc&per-page={num_works}"
        
        response = requests.get(works_url, timeout=10)
        response.raise_for_status()
        works = response.json().get("results", [])
        
        # Kombinera abstracts, titlar och concepts
        research_texts = []
        for work in works:
            text_parts = []
            
            # Lägg till titel
            if work.get("title"):
                text_parts.append(work["title"])
            
            # Lägg till abstract om det finns
            if work.get("abstract_inverted_index"):
                # Rekonstruera abstract från inverterat index
                abstract = reconstruct_abstract_from_inverted_index(work["abstract_inverted_index"])
                if abstract:
                    text_parts.append(abstract)
            
            # Lägg till concepts
            concepts = [c.get("display_name", "") for c in work.get("concepts", [])[:5]]
            if concepts:
                text_parts.extend(concepts)
            
            if text_parts:
                research_texts.append(" ".join(text_parts))
        
        return " ".join(research_texts) if research_texts else ""
        
    except requests.RequestException as e:
        st.warning(f"Kunde inte hämta publikationer för {author_id}: {e}")
        return ""

def reconstruct_abstract_from_inverted_index(inverted_index: Dict) -> str:
    """Rekonstruera abstract från OpenAlex inverterat index"""
    try:
        # Skapa en lista med rätt längd
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # Sortera efter position och skapa text
        word_positions.sort(key=lambda x: x[0])
        abstract = " ".join([word for pos, word in word_positions])
        
        # Begränsa längd för prestanda
        return abstract[:2000] if len(abstract) > 2000 else abstract
    except:
        return ""

def is_active_recently(author_json: Dict, years: int = 5) -> bool:
    """Kontrollera om författare varit aktiv nyligen"""
    current_year = datetime.datetime.now().year
    counts = author_json.get("counts_by_year", [])
    recent = [c for c in counts if c["year"] >= current_year - years + 1]
    total_recent_pubs = sum(c["works_count"] for c in recent)
    return total_recent_pubs > 0

def find_authors_by_concept(concept_id: str, per_page: int = 20) -> List[Dict]:
    """Hitta författare baserat på concept ID"""
    try:
        url = f"https://api.openalex.org/authors?filter=concepts.id:{concept_id}&sort=cited_by_count:desc&per-page={per_page}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.RequestException:
        return []

def find_authors_by_keyword(keyword: str, per_page: int = 20) -> List[Dict]:
    """Hitta författare baserat på keyword-sökning i publikationer"""
    try:
        url = f"https://api.openalex.org/works?search={keyword}&sort=cited_by_count:desc&per-page={per_page}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        works = response.json().get("results", [])
        
        authors = []
        seen_authors = set()
        
        for work in works:
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                author_id = author.get("id", "")
                
                if author_id and author_id not in seen_authors:
                    seen_authors.add(author_id)
                    authors.append({
                        "id": author_id,
                        "display_name": author.get("display_name", "Okänd"),
                        "matched_term": keyword,
                        "work_title": work.get("title", ""),
                        "cited_by_count": work.get("cited_by_count", 0)
                    })
                    
                    if len(authors) >= per_page:
                        break
            if len(authors) >= per_page:
                break
                
        return authors
    except requests.RequestException:
        return []

def calculate_weighted_score(author_data: Dict, call_text: str, selected_keywords: List[str], 
                           call_embedding: np.ndarray, max_citations: int) -> float:
    """Beräkna viktad matchningspoäng för en författare"""
    
    # 1. Semantisk likhet baserad på forskningsprofil
    research_profile = author_data.get('research_profile', '')
    if research_profile.strip():
        try:
            author_embedding = embed_model.encode(research_profile, convert_to_tensor=True, device="cpu")
            semantic_score = util.cos_sim(call_embedding, author_embedding)[0][0].item()
        except:
            semantic_score = 0.0
    else:
        semantic_score = 0.0
    
    # 2. Keyword-matchningspoäng
    matched_terms = set(term.lower() for term in author_data.get('matched_terms', []))
    selected_keywords_lower = set(kw.lower() for kw in selected_keywords)
    keyword_matches = len(matched_terms.intersection(selected_keywords_lower))
    keyword_score = min(keyword_matches / len(selected_keywords), 1.0) if selected_keywords else 0.0
    
    # 3. Impakt-poäng (normaliserad)
    citations = author_data.get('citations', 0)
    impact_score = min(citations / max(max_citations, 1), 1.0) if max_citations > 0 else 0.0
    
    # Viktad kombination
    final_score = (
        semantic_weight * semantic_score + 
        keyword_weight * keyword_score + 
        impact_weight * impact_score
    )
    
    return final_score, semantic_score, keyword_score, impact_score

# ---------- Förbättrad keyword-extraktion ----------
def extract_enhanced_keywords(text: str, num_keywords: int, ngram_range: int) -> List[str]:
    """Förbättrad keyword-extraktion med synonymer och domänspecifika termer"""
    
    # Grundläggande KeyBERT-extraktion
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, ngram_range),
        stop_words="english",
        top_n=num_keywords * 2,  # Ta ut fler för att sedan filtrera
        use_mmr=True,  # Använd Maximal Marginal Relevance för mångfald
        diversity=0.5
    )
    
    # Filtrera bort för generiska termer
    generic_terms = {
        'research', 'study', 'analysis', 'method', 'approach', 'development', 
        'investigation', 'application', 'project', 'work', 'data', 'results',
        'forskning', 'studie', 'analys', 'metod', 'utveckling', 'tillämpning',
        'projekt', 'arbete', 'data', 'resultat'
    }
    
    filtered_keywords = []
    for keyword, score in keywords:
        if (len(keyword) > 2 and 
            keyword.lower() not in generic_terms and 
            not keyword.replace(' ', '').isdigit()):
            filtered_keywords.append(keyword)
            
        if len(filtered_keywords) >= num_keywords:
            break
    
    return filtered_keywords

# ---------- Steg 1: Förbättrad sökord-extraktion ----------
if st.button("🔑 Extrahera sökord (Förbättrad)") or 'extracted_keywords' in st.session_state:
    if call_text.strip():
        if 'extracted_keywords' not in st.session_state:
            with st.spinner("Extraherar relevanta sökord..."):
                keywords = extract_enhanced_keywords(call_text, num_keywords, ngram_range)
                # Lägg till egna sökord först (om de inte redan finns)
                for custom_kw in custom_keywords:
                    if custom_kw and custom_kw not in keywords:
                        keywords.insert(0, custom_kw)
                st.session_state['extracted_keywords'] = keywords
                st.session_state['selected_keywords'] = keywords.copy()

        # Visa checkboxar
        st.write("### Identifierade sökord/fraser (välj de som är relevanta)")
        col1, col2 = st.columns(2)
        for i, kw in enumerate(st.session_state['extracted_keywords']):
            col = col1 if i % 2 == 0 else col2
            with col:
                checked = kw in st.session_state['selected_keywords']
                # Visa ikon för egna sökord
                icon = "🛠️ " if custom_keywords and kw in custom_keywords else "🔍 "
                val = st.checkbox(f"{icon}{kw}", value=checked, key=f"chk_{kw}")
                if val and kw not in st.session_state['selected_keywords']:
                    st.session_state['selected_keywords'].append(kw)
                elif not val and kw in st.session_state['selected_keywords']:
                    st.session_state['selected_keywords'].remove(kw)
    else:
        st.warning("Klistra in utlysningstext först.")

# ---------- Steg 2: Förbättrad forskarhämtning ----------
if st.button("✅ Hämta forskare (Förbättrad matchning)"):
    selected_keywords = st.session_state.get('selected_keywords', [])
    if not selected_keywords:
        st.warning("Välj minst ett sökord från steg 1.")
    else:
        with st.spinner("Hämtar forskare med förbättrad matchning..."):
            
            # Förencode utlysningstext för semantisk matchning
            call_embedding = embed_model.encode(call_text, convert_to_tensor=True, device="cpu")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Hitta concepts
            status_text.text("Söker forskningsconcept...")
            concepts = []
            for i, kw in enumerate(selected_keywords):
                try:
                    response = requests.get(f"https://api.openalex.org/concepts?search={kw}")
                    results = response.json().get("results", [])
                    if results:
                        concepts.append(results[0])
                except:
                    continue
                progress_bar.progress((i + 1) / (len(selected_keywords) * 2))
            
            # Hämta författare
            temp_authors = {}
            
            # Från concepts
            status_text.text("Hämtar forskare från concepts...")
            for i, concept in enumerate(concepts):
                authors = find_authors_by_concept(concept["id"], per_page=num_per_source)
                for author in authors:
                    author_id = author.get("id", "")
                    if author_id and author_id not in temp_authors:
                        temp_authors[author_id] = {
                            "matched_terms": set([concept.get("display_name", "")]),
                            "author_obj": author
                        }
                    elif author_id:
                        temp_authors[author_id]["matched_terms"].add(concept.get("display_name", ""))
                
                progress_bar.progress((len(selected_keywords) + i + 1) / (len(selected_keywords) * 2))
            
            # Från keywords
            for i, keyword in enumerate(selected_keywords):
                authors = find_authors_by_keyword(keyword, per_page=num_per_source)
                for author in authors:
                    author_id = author.get("id", "")
                    if author_id and author_id not in temp_authors:
                        temp_authors[author_id] = {
                            "matched_terms": set([keyword]),
                            "author_obj": author
                        }
                    elif author_id:
                        temp_authors[author_id]["matched_terms"].add(keyword)
            
            progress_bar.progress(1.0)
            status_text.text("Analyserar forskarprofiler...")
            
            # Hämta detaljerade profiler och beräkna poäng
            author_list = []
            max_citations = 0
            
            # Första gången igenom - hitta max citeringar för normalisering
            for author_id, info in temp_authors.items():
                details = fetch_author_details(author_id)
                if details:
                    citations = details.get("cited_by_count", 0)
                    max_citations = max(max_citations, citations)
            
            for i, (author_id, info) in enumerate(temp_authors.items()):
                details = fetch_author_details(author_id)
                
                if not details or not is_active_recently(details, years=5):
                    continue
                
                # Hämta forskningsprofil
                research_profile = get_author_research_profile(author_id)
                
                institutions = details.get("last_known_institutions", [])
                inst_name = institutions[0].get("display_name", "Okänd institution") if institutions else "Okänd institution"
                country = institutions[0].get("country_code", "Okänt land") if institutions else "Okänt land"
                
                works = details.get("works_count", 0)
                citations = details.get("cited_by_count", 0)
                
                # Skapa författardata för poängberäkning
                author_data = {
                    'research_profile': research_profile,
                    'matched_terms': list(info["matched_terms"]),
                    'citations': citations
                }
                
                # Beräkna viktad poäng
                final_score, semantic_score, keyword_score, impact_score = calculate_weighted_score(
                    author_data, call_text, selected_keywords, call_embedding, max_citations
                )
                
                author_list.append({
                    "Namn": details.get("display_name", "Okänd"),
                    "Institution": inst_name,
                    "Land": country,
                    "Publikationer": works,
                    "Citeringar": citations,
                    "Profil": author_id,
                    "Matched_terms": ", ".join(info["matched_terms"]),
                    "Viktad_poäng": round(final_score, 3),
                    "Semantisk_poäng": round(semantic_score, 3),
                    "Sökord_poäng": round(keyword_score, 3),
                    "Impakt_poäng": round(impact_score, 3),
                    "Forskningsprofil": research_profile[:200] + "..." if len(research_profile) > 200 else research_profile
                })
                
                # Uppdatera progress
                if i % 10 == 0:
                    status_text.text(f"Analyserat {i+1}/{len(temp_authors)} forskare...")
            
            progress_bar.empty()
            status_text.empty()
            
            df = pd.DataFrame(author_list)
            
            if df.empty:
                st.warning("Inga forskare hittades med dessa kriterier.")
            else:
                # Landfilter
                if include_countries:
                    df = df[df["Land"].isin(include_countries)]
                if exclude_countries:
                    df = df[~df["Land"].isin(exclude_countries)]
                
                if df.empty:
                    st.warning("Inga forskare kvar efter landfiltrering.")
                else:
                    # Sortera efter viktad poäng
                    df = df.sort_values("Viktad_poäng", ascending=False)
                    
                    # Visa statistik
                    st.success(f"Hittade {len(df)} aktiva forskare!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Medel semantisk poäng", f"{df['Semantisk_poäng'].mean():.3f}")
                    with col2:
                        st.metric("Medel sökord-poäng", f"{df['Sökord_poäng'].mean():.3f}")
                    with col3:
                        st.metric("Medel impakt-poäng", f"{df['Impakt_poäng'].mean():.3f}")
                    
                    # Visa tabell med förbättrade kolumner
                    display_df = df[[
                        "Namn", "Institution", "Land", "Publikationer", "Citeringar",
                        "Viktad_poäng", "Semantisk_poäng", "Sökord_poäng", "Impakt_poäng",
                        "Matched_terms", "Profil"
                    ]].copy()
                    
                    st.data_editor(
                        display_df,
                        column_config={
                            "Profil": st.column_config.LinkColumn("Profil", display_text="Öppna profil"),
                            "Viktad_poäng": st.column_config.NumberColumn("Viktad poäng", format="%.3f"),
                            "Semantisk_poäng": st.column_config.NumberColumn("Semantisk", format="%.3f"),
                            "Sökord_poäng": st.column_config.NumberColumn("Sökord", format="%.3f"),
                            "Impakt_poäng": st.column_config.NumberColumn("Impakt", format="%.3f"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Visa forskningsprofiler för topp 5
                    st.subheader("🧬 Forskningsprofiler (Topp 5)")
                    for i, (_, author) in enumerate(df.head(5).iterrows()):
                        with st.expander(f"{i+1}. {author['Namn']} - {author['Institution']}"):
                            st.write(f"**Viktad poäng:** {author['Viktad_poäng']}")
                            st.write(f"**Matchade termer:** {author['Matched_terms']}")
                            st.write(f"**Forskningsprofil:**")
                            st.write(author['Forskningsprofil'])

# ---------- Information om förbättringar ----------
with st.expander("ℹ️ Vad är nytt i denna version?"):
    st.markdown("""
    ### 🚀 Förbättringar för bättre matchningar:
    
    **1. Abstract-baserad semantisk matchning**
    - Analyserar forskares faktiska publikationer och abstracts
    - Mycket mer exakt matchning än bara namn + institution
    
    **2. Viktad scoring**
    - Kombinerar semantisk likhet, keyword-matchningar och forskningsimpakt
    - Anpassningsbara vikter i sidopanelen
    
    **3. Förbättrad keyword-extraktion**
    - Använder MMR (Maximal Marginal Relevance) för mer varierade sökord
    - Filtrerar bort generiska termer
    
    **4. Prestanda-optimeringar**
    - Caching av API-anrop
    - Progress bars för användarfeedback
    - Bättre felhantering
    
    **5. Utökad analys**
    - Visar forskningsprofiler för toppresultat
    - Separata poäng för olika matchningstyper
    """)







