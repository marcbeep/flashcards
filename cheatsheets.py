import streamlit as st
from pathlib import Path

# Configure the Streamlit page
st.set_page_config(page_title="Markdown Viewer", page_icon="📝", layout="centered")

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


def _convert_latex_delimiters(text: str) -> str:
    """Convert LaTeX delimiters to Streamlit-compatible format."""
    # Convert display math mode
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")

    # Convert inline math mode
    text = text.replace("\\(", "$")
    text = text.replace("\\)", "$")

    # Handle common LaTeX commands that might need escaping
    text = text.replace("\\mathbf{", "\\mathbf{")  # Ensure bold math works
    text = text.replace("\\sum_", "\\sum\\limits_")  # Better sum limits

    return text


def _load_markdown(path: Path) -> str:
    """Load markdown content from a file."""
    content = path.read_text(encoding="utf-8")
    return _convert_latex_delimiters(content)


with st.sidebar:
    st.header("Cheatsheets")
    st.markdown(
        """Enter the directory: `bi/cheatsheets`, `ci/cheatsheets`, or `dm/cheatsheets`"""
    )
    DIR = Path(st.text_input("Markdown directory", value="."))

files = sorted(DIR.glob("*.md"))
if not files:
    st.sidebar.error("No markdown files found in the specified directory.")
else:
    file_sel = st.sidebar.selectbox(
        "Choose a file", files, format_func=lambda p: p.name
    )
    content = _load_markdown(file_sel)
    st.markdown(content)
