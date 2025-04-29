import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path
import random

# Configure the Streamlit page with LaTeX support
st.set_page_config(page_title="Flashcards", page_icon="📚", layout="centered")

# Enable LaTeX support with better defaults
st.markdown(
    """
<style>
.katex-html {
    text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)


def _texify(text: str) -> str:
    # Handle escaped newlines by converting them to actual newlines
    text = text.replace("\\n", "\n")

    # Ensure display math has proper spacing and centering
    text = text.replace("\\[", "\n\n$$\n")
    text = text.replace("\\]", "\n$$\n\n")

    # Ensure inline math has proper spacing but stays inline
    text = text.replace("\\(", "$$")
    text = text.replace("\\)", "$$")

    # Handle common LaTeX commands that might need escaping
    text = text.replace("\\mathbf{", "\\mathbf{")  # Ensure bold math works
    text = text.replace("\\top", "\\top ")  # Add space after \top
    text = text.replace("\\sum_", "\\sum\\limits_")  # Better sum limits

    return text


def _load_cards(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"q": _texify(c["q"]), "a": _texify(c["a"])} for c in data]


def _card_viewer(cards):
    if not cards:
        st.error("No cards found in this file.")
        return

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "show" not in st.session_state:
        st.session_state.show = False
    if "current_deck" not in st.session_state:
        st.session_state.current_deck = cards

    # Create a single row for all navigation buttons
    nav_cols = st.columns([1, 1, 1, 1])
    with nav_cols[0]:
        if st.button("⬅️ Previous"):
            st.session_state.idx = (st.session_state.idx - 1) % len(
                st.session_state.current_deck
            )
            st.session_state.show = False
    with nav_cols[1]:
        if st.button("Show / Hide Answer"):
            st.session_state.show = not st.session_state.show
    with nav_cols[2]:
        if st.button("Next ➡️", key="next_btn"):
            st.session_state.idx = (st.session_state.idx + 1) % len(
                st.session_state.current_deck
            )
            st.session_state.show = False
    with nav_cols[3]:
        st.caption(
            f"Card {st.session_state.idx + 1} / {len(st.session_state.current_deck)}"
        )

    # Display the card content
    card = st.session_state.current_deck[st.session_state.idx]
    st.markdown(f"### Q{st.session_state.idx + 1}: {card['q']}")
    if st.session_state.show:
        st.markdown(card["a"])


with st.sidebar:
    st.header("Flashcards")
    st.markdown("""Enter the directory: `bi/cards`, `ci/cards`, or `dm/cards`""")
    DIR = Path(st.text_input("Flashcard directory", value="."))
    DO_SHUFFLE = st.button("🔀 Shuffle deck")

files = sorted(DIR.glob("*.json"))
if not files:
    st.sidebar.error("No JSON files found in the specified directory.")
else:
    file_sel = st.sidebar.selectbox("Choose a set", files, format_func=lambda p: p.name)
    deck = _load_cards(file_sel)
    if DO_SHUFFLE:
        random.shuffle(deck)
        st.session_state.current_deck = deck
        st.session_state.idx, st.session_state.show = 0, False
    _card_viewer(deck)
