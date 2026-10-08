# NewsRAG

Ask questions about **609 news articles from late 2023** (TechCrunch, The Verge,
Sporting News, and more). Many questions need evidence from **several articles**,
and every answer cites the excerpts it is based on.

- Click a **Source N** link to read the excerpt behind a claim.
- Use the ⚙️ settings to switch between **multihop** (splits the question into one
  search per outlet or event; best for comparisons) and **baseline** (one search; faster).
- Questions in **Thai** work too: the search is multilingual.
- If the articles do not contain the answer, the assistant says
  *"Insufficient information."* instead of guessing.

Dataset: [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG)
(Tang & Yang, 2024), ODC-BY license.
