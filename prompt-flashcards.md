ROLE:
Create flashcards from lecture content.

TASK:
Write Basic (Q→A) cards in JSON format.

RULES:

- Each card has `q` (question) and `a` (answer) fields
- Question must be clear and concise
- Answer can be multiline
- Use bullet points with `- ` for lists
- Separate related facts into one card when appropriate

MATH & CODE:

- Inline math: `\\( ... \\)` (will be rendered as `$$ ... $$`)
- Display math: `\\[ ... \\]` (will be rendered as `$$\n...\n$$`)
- Code: `` `...` ``
- Use `\\mathbf{...}` for bold math
- Use `\\sum\\limits_` for better sum limits
- Add space after `\\top`

OUTPUT:
JSON array of objects with `q` and `a` fields.
Example:

[
{
"q": "State the Hebbian learning principle, its weight update, and a stabilization variant.",
"a": "- Principle: \"Neurons that fire together, wire together.\"\n\n- Basic update:\n\\[ \\Delta w_{kj}(n) = \\eta\\, y_k(n)\\, x_j(n) \\]\n- Limitation: unbounded growth (synaptic saturation).\n- **Covariance hypothesis** stabilizes weights:\n\\[ \\Delta w_{kj}(n) = \\eta\\, (y_k - \\bar{y})(x_j - \\bar{x}) \\]"
},
{
"q": "How does competitive learning work and which formulas govern it?",
"a": "- Neurons compete; only the **winner** fires:\n\\[ y_k=\\begin{cases}1 & v_k > v_j,\\ \\forall j\\neq k\\\\0&\\text{otherwise}\\end{cases} \\]\n\n- Winner's weights move toward the input:\n\\[ \\Delta w_{ki} = \\eta (x_i - w_{ki}) \\]\n- Produces specialization for clustering and Self-Organizing Maps."
}
]
