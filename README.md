# Uganda Health Assistant (E2E ChatBot)

## 🚀 Project Overview
This project builds a complete functional health assistant chatbot for Uganda context using free open-source LLMs and simple local knowledge retrieval. It includes:

- Streamlit UI for conversational chat
- Open-source model `google/flan-t5-small` via `transformers`
- Knowledge-base retrieval from local JSON
- LLM Ops evaluation and configuration modules
- Beginner-friendly step-by-step documentation

## ✅ Why this design?
1. **Free and open-source**: Uses Transformers and Sentence Transformers with free models.
2. **Simple deployment**: Streamlit app can deploy to Hugging Face Spaces directly.
3. **Context-aware**: Retrieves relevant local health knowledge from a curated JSON dataset.
4. **Safe defaults**: The prompt clearly instructs safe, local, health-focused responses.

## 📁 Repo Structure
```
app.py
requirements.txt
src/health_assistant.py
src/llm_ops.py
data/uganda_health_knowledge.json
README.md
```

## 🧠 How it works (LLM Ops approach)
1. In `src/health_assistant.py`, we load knowledge items and compute embeddings with SentenceTranformers.
2. We build a retrieval function to find relevant health snippets from user questions.
3. We pass retrieved context into a prompt for text2text-generation with Flan-T5.
4. `src/llm_ops.py` wraps chat calls and makes eval methods reusable.

## ▶️ Run locally
1. Create venv:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
2. Run app:
```bash
streamlit run app.py
```
3. Open `http://localhost:8501`

## 🧪 Advanced Methods Used
- **Prompt engineering**: explicit system prompt for friendly Uganda-health tone and safe instructions.
- **Retrieval-augmented generation**: local knowledge retrieval before model generation.
- **LLM Ops eval**: `LLMManager.run_eval()` can run many sample questions for quality checks.
- **Config-driven model parameters**: temperature and max length are adjustable in the sidebar.

## 📝 Deploy to Hugging Face Spaces (Streamlit)
1. Create a new Space on Hugging Face, choose Streamlit.
2. Push this repo to GitHub.
3. Connect the Space to the GitHub repo.
4. Add `requirements.txt` and `app.py` (already there); the Space installs dependencies.
5. If huggingface needs model weights, ensure internet access is allowed in your Space.

> Tip: In Spaces, you can add `HF_HUB_TOKEN` as secret if using private model; here we use public free model.

## 💡 Why this is beginner-friendly
- All code is explicit and short.
- We avoid external paid cloud APIs.
- The README explains each file and step with concrete commands.
- You can extend the knowledge JSON easily by adding more Q/A pairs.

## 🧩 Next improvements (for perfection)
- Add a formal evaluation script `scripts/evaluate.py` to log model responses.
- Add user identity and safe disclaimers before health advice.
- Add data versioning (store `data/health_knowledge_v1.json` etc.).
- Add UI for adding local knowledge from an admin form.

## 🧾 GitHub push steps
```bash
git add .
git commit -m "Add functional Uganda health assistant chatbot"
git push origin main
```

## 🧭 Quick usage example
1. Ask: "How do I manage fever in my child?"
2. Model returns actionable advice + go to clinic guidance.
3. Ask: "Where can I get HIV testing in Kampala?" and get referral style answer.

---

If you want me to next add evaluation metrics and a Jinja-based email report for LLM Ops Test results, say "Next: add evaluation pipeline" and I’ll implement it directly. 
