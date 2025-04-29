import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path
import random

# Configure the Streamlit page with LaTeX support
st.set_page_config(page_title="JSON Flashcards", page_icon="📚", layout="centered")

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

    prev_c, mid_c, next_c = st.columns([1, 2, 1])
    with prev_c:
        if st.button("⬅️ Previous"):
            st.session_state.idx = (st.session_state.idx - 1) % len(cards)
            st.session_state.show = False
    with mid_c:
        if st.button("Show / Hide Answer"):
            st.session_state.show = not st.session_state.show
    with next_c:
        if st.button("Next ➡️", key="next_btn"):
            st.session_state.idx = (st.session_state.idx + 1) % len(cards)
            st.session_state.show = False

    card = cards[st.session_state.idx]
    st.markdown(f"### Q{st.session_state.idx + 1}: {card['q']}")
    if st.session_state.show:
        st.markdown(card["a"])
    st.caption(f"Card {st.session_state.idx + 1} / {len(cards)}")


# Main
st.title("📚 JSON Flashcards")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        """Pick where your decks live:
        1. **Directory** – folder that contains your `.json` files
        2. **Choose a set** – select the file to study
        3. **Shuffle** – randomize order anytime
        """
    )
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
        st.session_state.idx, st.session_state.show = 0, False
    _card_viewer(deck)
