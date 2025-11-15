import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000"  # use the same host as uvicorn logs

st.set_page_config(page_title="RAG Pipeline Optimizer", layout="wide")

st.title("🧪 RAG Pipeline Optimizer (MLOps Project)")

st.sidebar.header("Backend")
st.sidebar.write(f"Backend: `{BACKEND_URL}`")

st.header("1. Upload documents")
uploaded_files = st.file_uploader(
    "Upload HR policies / docs (.txt, .md, .pdf)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)

# Manual indexing button
if st.button("Index documents", disabled=not uploaded_files):
    files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
    with st.spinner("Indexing in all pipelines..."):
        try:
            r = requests.post(f"{BACKEND_URL}/upload_docs", files=files)
        except Exception as e:
            st.error(f"Failed to call backend /upload_docs: {e}")
        else:
            if r.status_code == 200:
                st.success(r.json().get("message", "Indexed documents."))
            else:
                st.error(f"Backend error: {r.status_code} – {r.text}")

st.header("2. Ask a question")

question = st.text_input(
    "Question",
    placeholder="e.g. When does COBRA coverage begin for terminated employees?",
)
top_k = st.slider("Top K retrieved chunks", 2, 10, 5)

if st.button("Run RAG pipelines", disabled=not question):
    # 🔐 Safety: ensure docs are indexed right before asking
    if uploaded_files:
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        with st.spinner("Ensuring documents are indexed in all pipelines..."):
            try:
                r_index = requests.post(f"{BACKEND_URL}/upload_docs", files=files)
            except Exception as e:
                st.error(f"Failed to call backend /upload_docs: {e}")
                st.stop()
            else:
                if r_index.status_code == 200:
                    st.info(
                        r_index.json().get(
                            "message",
                            "Re-indexed documents before running pipelines.",
                        )
                    )
                else:
                    st.error(
                        f"Backend error during indexing: {r_index.status_code} – {r_index.text}"
                    )
                    st.stop()
    else:
        st.warning("Please upload at least one document before asking a question.")
        st.stop()

    # Now call /ask
    with st.spinner("Running 4 pipelines + evaluator..."):
        try:
            r = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question, "top_k": top_k},
            )
        except Exception as e:
            st.error(f"Failed to call backend /ask: {e}")
        else:
            if r.status_code != 200:
                st.error(f"Backend error: {r.status_code} – {r.text}")
            else:
                data = r.json()
                st.success(
                    f"Best pipeline for this question: "
                    f"{data['best_pipeline_id']} – {data['best_pipeline_name']}"
                )

                # Table of scores
                rows = []
                for p in data["pipelines"]:
                    s = p["scores"]
                    rows.append(
                        {
                            "Pipeline": f"{p['pipeline_id']} – {p['pipeline_name']}",
                            "Accuracy": s["accuracy"],
                            "Relevance": s["relevance"],
                            "Cost score": s["cost"],
                            "Overall": s["overall"],
                        }
                    )
                df = pd.DataFrame(rows)

                st.subheader("Scores per pipeline")
                st.dataframe(df, use_container_width=True)

                st.subheader("Bar chart (overall score)")
                st.bar_chart(df.set_index("Pipeline")["Overall"])

                # Show answers
                st.subheader("Pipeline answers")
                for p in data["pipelines"]:
                    with st.expander(
                        f"Pipeline {p['pipeline_id']} – {p['pipeline_name']}"
                    ):
                        st.markdown(f"**Overall score:** {p['scores']['overall']}")
                        st.markdown("**Answer:**")
                        st.write(p["answer"])
                        st.markdown("**Top retrieved chunks:**")
                        for i, ch in enumerate(p["retrieved_chunks"], start=1):
                            st.markdown(f"*Chunk {i}:*")
                            st.code(ch[:1000])