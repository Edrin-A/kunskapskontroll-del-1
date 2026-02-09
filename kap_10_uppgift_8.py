import streamlit as st
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
import unicodedata
from dotenv import load_dotenv

load_dotenv()

def fix_swedish_chars(text):
    """Fixar vanliga svenska tecken som kan bli felkodade från PDF."""
    # Hantera mellanrum mellan diakritiska tecken och bokstäver
    import re

    # Med eller utan mellanrum
    patterns = [
        (r'˚\s*a', 'å'), (r'¨\s*a', 'ä'), (r'¨\s*o', 'ö'),
        (r'˚\s*A', 'Å'), (r'¨\s*A', 'Ä'), (r'¨\s*O', 'Ö'),
        (r'´\s*e', 'é'), (r'´\s*E', 'É'),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    # Normalisera först med NFKD för att separera, sedan NFC för att kombinera
    text = unicodedata.normalize('NFKD', text)
    text = unicodedata.normalize('NFC', text)

    return text

# API-koppling
api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

# Chunking
def chunk_text(text, chunk_size=1000, overlap=200):
    """Delar upp text i mindre bitar med överlapp."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# YouTube funktioner
def extract_video_id(url):
    """Extraherar video ID från olika YouTube URL-format."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_url):
    """Hämtar transkript från en YouTube video."""
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Ogiltig YouTube-URL")

    try:
        # Skapa API instans
        ytt_api = YouTubeTranscriptApi()

        # Försök hämta transkript med olika språk (svenska först, sedan engelska)
        languages_to_try = ['sv', 'en', 'en-US', 'en-GB']
        transcript = None

        for lang in languages_to_try:
            try:
                transcript = ytt_api.fetch(video_id, languages=[lang])
                break
            except:
                continue

        # Om inget språk fungerade, försök utan språkspecifikation
        if transcript is None:
            transcript = ytt_api.fetch(video_id)

        # Kombinera all text från snippets
        full_text = " ".join([snippet.text for snippet in transcript.snippets])
        return full_text
    except Exception as e:
        raise ValueError(f"Kunde inte hämta transkript: {str(e)}")

# Embeddings
def create_embeddings(chunks, model="gemini-embedding-001", task_type="SEMANTIC_SIMILARITY"):
    """Skapar embeddings för en lista av chunks. Hanterar API gränsen på 100 per batch."""
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        all_embeddings.extend(response.embeddings)

    return all_embeddings

def create_single_embedding(text, model="gemini-embedding-001", task_type="SEMANTIC_SIMILARITY"):
    """Skapar embedding för en enskild text (t.ex. en fråga)."""
    response = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return response.embeddings[0].values

# Semantisk sökning
# Score kommer från cosine similarity mellan embeddings, och visar hur semantiskt lik en textbit är din fråga. Ju högre procent, desto mer relevant är chunken!
def cosine_similarity(vec1, vec2):
    """Beräknar cosine similarity mellan två vektorer."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, chunks, embeddings, k=5, relevance_threshold=0.5):
    """Söker efter de mest relevanta chunks baserat på frågan."""
    query_embedding = create_single_embedding(query)
    similarity_scores = []

    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(query_embedding, chunk_embedding.values)
        similarity_scores.append((i, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Filtrera på relevans tröskel och ta max k resultat
    filtered_results = [(i, score) for i, score in similarity_scores if score >= relevance_threshold]
    top_results = filtered_results[:k]

    return [(chunks[i], score) for i, score in top_results]

# Generera svar
SYSTEM_PROMPT = """Du är en hjälpsam assistent som svarar på frågor baserat
på den kontext som ges. Du har tillgång till tidigare konversation för att
kunna hantera uppföljningsfrågor.

Regler:
- Basera alltid dina svar på informationen i kontexten
- Du FÅR kombinera och dra slutsatser från olika delar av kontexten
- När du kombinerar information, var tydlig med vad som kommer från vilken del
- Om svaret verkligen inte finns i kontexten, säg "Jag kan inte hitta svaret på den frågan i dokumentet"
- Hitta INTE PÅ information som inte finns i kontexten
- Var tydlig och strukturerad i dina svar
- Citera gärna relevanta delar från kontexten när det passar
- Svara på samma språk som frågan ställs på
- Vid uppföljningsfrågor (t.ex. "kan du förklara mer om det?"), använd tidigare konversation för kontext
- Om användaren säger "det", "detta", etc., referera till tidigare diskussion
"""

def generate_response(query, chunks, embeddings, conversation_history=None, model="gemini-2.0-flash", relevance_threshold=0.5, max_chunks=5):
    """Genererar svar baserat på semantisk sökning i dokumentet."""
    # Hämta relevanta chunks med scores och tröskel
    relevant_chunks = semantic_search(query, chunks, embeddings, k=max_chunks, relevance_threshold=relevance_threshold)

    # Kontrollera om vi har bra källor
    low_relevance_warning = False
    if not relevant_chunks:
        # Inga chunks över threshold, använd top 3 ändå men varna
        relevant_chunks = semantic_search(query, chunks, embeddings, k=3, relevance_threshold=0.0)
        low_relevance_warning = True
    elif relevant_chunks and relevant_chunks[0][1] < 0.65:
        # Bästa träffen är under 65%, måttlig varning
        low_relevance_warning = True

    # Bygg kontext från chunks
    context = "\n\n".join([chunk for chunk, score in relevant_chunks])

    # Bygg conversation history om det finns
    history_text = ""
    if conversation_history and len(conversation_history) > 0:
        # Ta de senaste 4 meddelandena (2 Q&A par)
        recent_history = conversation_history[-4:]
        history_parts = []
        for msg in recent_history:
            role = "Användare" if msg["role"] == "user" else "Assistent"
            history_parts.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_parts)

    # Skapa user prompt med eller utan historik
    if history_text:
        user_prompt = f"""Tidigare konversation:
{history_text}

Nuvarande fråga: {query}

Här är relevant kontext från dokumentet:
{context}"""
    else:
        user_prompt = f"Frågan är: {query}\n\nHär är kontexten:\n{context}"

    # Generera svar med lägre temperatur för mer faktabaserade svar
    response = client.models.generate_content(
        model=model,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,  # Låg temperatur = mer faktabaserad
            top_p=0.8
        ),
        contents=user_prompt
    )

    return response.text, relevant_chunks, low_relevance_warning

def generate_example_questions(chunks, num_questions=4):
    """Genererar exempel frågor baserat på dokumentets innehåll."""
    # Ta ett sample av chunks för att få en överblick
    sample_size = min(10, len(chunks))
    sample_chunks = chunks[::len(chunks)//sample_size][:sample_size]
    sample_text = "\n\n".join(sample_chunks)

    # Begränsa längden
    if len(sample_text) > 5000:
        sample_text = sample_text[:5000]

    prompt = f"""Baserat på följande textutdrag, generera {num_questions} intressanta och relevanta frågor som någon skulle kunna ställa om innehållet.

Regler:
- Frågorna ska vara specifika och relevanta för texten
- Variera typen av frågor (vad, hur, varför, etc.)
- Gör frågorna koncisa (max 10-15 ord)
- Skriv ENDAST frågorna, en per rad, utan numrering eller punkter
- Svara på svenska

Text:
{sample_text}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=genai.types.GenerateContentConfig(temperature=0.7),
            contents=prompt
        )

        # Dela upp i individuella frågor
        questions = [q.strip() for q in response.text.strip().split('\n') if q.strip() and '?' in q]
        return questions[:num_questions]
    except Exception as e:
        # Fallback om något går fel
        return [
            "Vad handlar dokumentet om?",
            "Kan du sammanfatta huvudpunkterna?",
            "Vilka är de viktigaste begreppen?",
            "Finns det några exempel i texten?"
        ]

# Streamlit UI
st.set_page_config(page_title="RAG Chattbot", page_icon="📄")
st.title("RAG Chattbot")
st.write("Ladda upp en PDF eller klistra in en YouTube länk och ställ frågor!")

# Sidebar för källa
with st.sidebar:
    st.header("Välj källa")
    source_type = st.radio("Typ av källa:", ["PDF", "YouTube"], horizontal=True)

    # Avancerade inställningar
    st.divider()
    with st.expander("Sökinställningar"):
        relevance_threshold = st.slider(
            "Relevans tröskel",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum relevans för att inkludera en källa. Högre värde = strängare filter."
        )
        max_chunks = st.slider(
            "Max antal källor",
            min_value=1,
            max_value=10,
            value=5,
            help="Max antal textdelar att använda för att svara."
        )

    st.divider()

    if source_type == "PDF":
        uploaded_files = st.file_uploader("Välj PDF filer", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            # Skapa en unik nyckel baserat på alla filnamn
            file_names = sorted([f.name for f in uploaded_files])
            files_key = "|".join(file_names)

            # Kolla om det är nya filer
            if "current_source" not in st.session_state or st.session_state.current_source != files_key:
                with st.spinner(f"Bearbetar {len(uploaded_files)} PDF fil(er)..."):
                    # Rensa gamla data
                    st.session_state.chunks = None
                    st.session_state.embeddings = None
                    st.session_state.messages = []
                    st.session_state.current_source = files_key
                    st.session_state.source_type = "PDF"
                    st.session_state.file_names = file_names

                    # Läs in alla PDF filer och samla metadata
                    all_text = ""
                    total_pages = 0
                    total_size = 0
                    file_info = []

                    for uploaded_file in uploaded_files:
                        reader = PdfReader(uploaded_file)
                        pages = len(reader.pages)
                        total_pages += pages
                        total_size += uploaded_file.size

                        file_text = ""
                        for page in reader.pages:
                            extracted = page.extract_text() or ""
                            # Fixa svenska tecken och normalisera
                            file_text += fix_swedish_chars(extracted)

                        file_info.append({
                            "name": uploaded_file.name,
                            "pages": pages,
                            "size": uploaded_file.size
                        })

                        all_text += file_text + "\n\n"

                    # Spara metadata
                    st.session_state.file_info = file_info
                    st.session_state.total_pages = total_pages
                    st.session_state.total_size = total_size
                    st.session_state.total_chars = len(all_text)
                    st.session_state.total_words = len(all_text.split())

                    # Chunka texten
                    st.session_state.chunks = chunk_text(all_text)

                    # Skapa embeddings
                    st.session_state.embeddings = create_embeddings(st.session_state.chunks)

                    # Generera exempel frågor
                    with st.spinner("Genererar exempel frågor..."):
                        st.session_state.example_questions = generate_example_questions(st.session_state.chunks)

                st.success(f"{len(uploaded_files)} PDF fil(er) laddade!")

            # Visa info om dokumenten
            st.divider()
            st.subheader("Dokumentinfo")

            # Visa varje fil med detaljer
            for info in st.session_state.get("file_info", []):
                size_kb = info["size"] / 1024
                st.markdown(f"**{info['name']}**")
                st.caption(f"{info['pages']} sidor | {size_kb:.1f} KB")

            # Sammanfattning
            st.divider()
            st.markdown("**Sammanfattning**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Totalt sidor", st.session_state.get("total_pages", 0))
                st.metric("Chunks", len(st.session_state.chunks))
            with col2:
                st.metric("Ord", f"{st.session_state.get('total_words', 0):,}".replace(",", " "))
                size_mb = st.session_state.get("total_size", 0) / (1024 * 1024)
                st.metric("Storlek", f"{size_mb:.2f} MB")
        else:
            if "current_source" in st.session_state and st.session_state.get("source_type") == "PDF":
                st.session_state.clear()

    else:  # YouTube
        youtube_url = st.text_input("Klistra in YouTube länk:", placeholder="https://www.youtube.com/watch?v=...")

        if st.button("Ladda video", type="primary"):
            if youtube_url:
                # Kolla om det är en ny URL
                if "current_source" not in st.session_state or st.session_state.current_source != youtube_url:
                    with st.spinner("Hämtar transkript från YouTube..."):
                        try:
                            # Rensa gamla data
                            st.session_state.chunks = None
                            st.session_state.embeddings = None
                            st.session_state.messages = []
                            st.session_state.current_source = youtube_url
                            st.session_state.source_type = "YouTube"

                            # Hämta transkript
                            text = get_youtube_transcript(youtube_url)

                            # Spara metadata om videon
                            video_id = extract_video_id(youtube_url)
                            st.session_state.video_id = video_id
                            st.session_state.transcript_chars = len(text)
                            st.session_state.transcript_words = len(text.split())

                            # Chunka texten
                            st.session_state.chunks = chunk_text(text)

                            # Skapa embeddings
                            st.session_state.embeddings = create_embeddings(st.session_state.chunks)

                            # Generera exempel frågor
                            with st.spinner("Genererar exempel-frågor..."):
                                st.session_state.example_questions = generate_example_questions(st.session_state.chunks)

                            st.success("YouTube-video laddad!")
                        except ValueError as e:
                            st.error(str(e))
            else:
                st.warning("Ange en YouTube länk först!")

        # Visa info om videon om den är laddad
        if st.session_state.get("source_type") == "YouTube" and st.session_state.get("chunks"):
            st.divider()
            st.subheader("Videoinfo")

            # Video-ID med länk
            video_id = st.session_state.get("video_id", "")
            st.markdown(f"**Video-ID:** `{video_id}`")

            # Thumbnail
            if video_id:
                st.image(f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg", use_container_width=True)

            # Statistik
            st.divider()
            st.markdown("**Transkript-statistik**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ord", f"{st.session_state.get('transcript_words', 0):,}".replace(",", " "))
                st.metric("Chunks", len(st.session_state.chunks))
            with col2:
                chars = st.session_state.get("transcript_chars", 0)
                st.metric("Tecken", f"{chars:,}".replace(",", " "))
                # Uppskattad lästid (ca 200 ord/min)
                words = st.session_state.get("transcript_words", 0)
                read_time = max(1, words // 200)
                st.metric("Lästid", f"~{read_time} min")

# Initiera chatthistorik
if "messages" not in st.session_state:
    st.session_state.messages = []

# Visa chatthistorik
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Källor som användes"):
                for i, (chunk, score) in enumerate(message["sources"]):
                    # Färgkodning baserat på relevans
                    if score >= 0.75:
                        color = "🟢"  # Grön - Hög relevans
                    elif score >= 0.60:
                        color = "🟡"  # Gul - Medel relevans
                    else:
                        color = "🔴"  # Röd - Låg relevans

                    st.write(f"{color} **Chunk {i+1}** (relevans: {score:.2%})")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()

# Visa exempel frågor om det inte finns några meddelanden ännu
if st.session_state.get("chunks") is not None and len(st.session_state.messages) == 0:
    if "example_questions" in st.session_state and st.session_state.example_questions:
        st.markdown("### 💭 Förslag på frågor")
        st.caption("Klicka på en fråga för att ställa den:")

        # Visa frågor i kolumner (2 per rad)
        cols = st.columns(2)
        for idx, question in enumerate(st.session_state.example_questions):
            col = cols[idx % 2]
            with col:
                if st.button(question, key=f"example_q_{idx}", use_container_width=True):
                    # Sätt frågan som nästa input
                    st.session_state.next_question = question
                    st.rerun()

# Chatt input
if st.session_state.get("chunks") is not None:
    source_label = "videon" if st.session_state.get("source_type") == "YouTube" else "dokumentet"

    # Visa alltid chat input
    user_input = st.chat_input(f"Ställ en fråga om {source_label}...")

    # Kolla om det finns en fråga från exempel knapp
    prompt = None
    if "next_question" in st.session_state:
        prompt = st.session_state.next_question
        del st.session_state.next_question
    elif user_input:
        prompt = user_input

    if prompt:
        # Lägg till användarens meddelande
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generera svar
        with st.chat_message("assistant"):
            with st.spinner("Tänker..."):
                response, sources, low_relevance = generate_response(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    conversation_history=st.session_state.messages,
                    relevance_threshold=relevance_threshold,
                    max_chunks=max_chunks
                )

            # Visa varning om låg relevans
            if low_relevance:
                st.warning("**Låg relevans**: Jag hittade ingen starkt relevant information för din fråga. Svaret kan vara osäkert.")

            st.markdown(response)

            # Visa källor med färgkodning
            with st.expander("Källor som användes"):
                for i, (chunk, score) in enumerate(sources):
                    # Färgkodning baserat på relevans
                    if score >= 0.75:
                        color = "🟢"  # Grön - Hög relevans
                    elif score >= 0.60:
                        color = "🟡"  # Gul - Medel relevans
                    else:
                        color = "🔴"  # Röd - Låg relevans

                    st.write(f"{color} **Chunk {i+1}** (relevans: {score:.2%})")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()

        # Spara assistentens svar
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
else:
    st.info("Välj en källa i sidofältet (PDF eller YouTube) för att börja!") 




# Förbättringar:
# 1. Lagt till stöd för YouTube transkript som källa.
# 2. Visar metadata om uppladdade PDF filer och YouTube videor.
# 3. Hanterar flera PDF filer samtidigt.
# 4. Fixade svenska tecken vid källor från ai svar.
# 5. Konversation historia, hanterar uppföljningsfrågor genom att inkludera tidigare konversation.
# 6. Relevans tröskel, filtrerar bort irrelevanta källor och varnar vid låg relevans.
# 7. Färgkodade relevans scores, visuell indikator för källornas relevans.
# 8. Justerbara sökinställningar. Användaren kan ändra relevans tröskel och antal källor.
# 9. Temperatur kontroll (0.3). Mer faktabaserade och mindre "kreativa" svar.
# 10. Exempel frågor, AI genererar automatiskt 4 relevanta frågor baserat på innehållet.

# Frågor:
# 1. Hur kan jag bekämpa prokrastinering när jag ska lära mig om AI?
# 2. Vilka tips finns om att läsa kurslitteratur och hur kan jag använda dem för att förstå RAG-system?