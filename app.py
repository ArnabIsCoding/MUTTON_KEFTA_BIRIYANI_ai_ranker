
import streamlit as st
import pandas as pd
import tempfile
import os
import time

st.set_page_config(page_title="Redrob AI Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Redrob AI Candidate Ranker")
st.markdown("""
**Intelligent Candidate Discovery & Ranking** for Senior AI Engineer roles.

Upload a candidates file (`.json` or `.jsonl`) and the ranker will score and rank
the top 100 candidates using an 9-group scoring system.
""")

uploaded = st.file_uploader("Upload candidates file", type=["json", "jsonl"])

if uploaded:
    if "tmp_path" not in st.session_state or st.session_state.get("tmp_upload_name") != uploaded.name:
        suffix = ".jsonl" if uploaded.name.endswith(".jsonl") else ".json"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
            tmp.write(uploaded.read())
            st.session_state["tmp_path"] = tmp.name
            st.session_state["tmp_upload_name"] = uploaded.name

    tmp_path = st.session_state["tmp_path"]

    if st.button("🚀 Run Ranker", type="primary"):
        with st.spinner("Running candidate ranking pipeline..."):
            start = time.time()

            from src.parser import parse_all_candidates
            from src.features import compute_all_features
            from src.scorer import compute_final_scores, select_top_n
            from src.reasoning import generate_reasoning

            candidates = parse_all_candidates(tmp_path)
            st.info(f"Loaded {len(candidates)} candidates.")

            from datetime import datetime
            valid_dates = []
            for c in candidates:
                d_str = c.get("last_active_date")
                if d_str:
                    try:
                        valid_dates.append(datetime.strptime(d_str, "%Y-%m-%d").date())
                    except (ValueError, TypeError):
                        pass
            from src.config import REFERENCE_DATE
            dynamic_ref_date = max(valid_dates) if valid_dates else REFERENCE_DATE

            features_list = [compute_all_features(c, tfidf_lookup={}, reference_date=dynamic_ref_date) for c in candidates]

            scored_df = compute_final_scores(features_list)
            top_df = select_top_n(scored_df, n=100)

            cand_lookup = {c["candidate_id"]: c for c in candidates}
            feat_lookup = {f["candidate_id"]: f for f in features_list}
            reasoning_list = []
            for _, row in top_df.iterrows():
                cid = row["candidate_id"]
                reasoning = generate_reasoning(
                    cand_lookup.get(cid, {}),
                    feat_lookup.get(cid, {}),
                    int(row["rank"]),
                )
                reasoning_list.append(reasoning)
            top_df = top_df.copy()
            top_df["reasoning"] = reasoning_list

            elapsed = time.time() - start
            st.success(f"✅ Ranking complete in {elapsed:.1f}s")

        st.subheader("📊 Top 100 Candidates")
        display_cols = ["rank", "candidate_id", "final_score",
                        "group_a_title", "group_b_desc", "group_c_skills",
                        "group_d_gate", "group_g_behavioral", "group_h_honeypot"]
        st.dataframe(top_df[display_cols], use_container_width=True, height=400)

        st.subheader("📈 Score Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top Score", f"{top_df['final_score'].iloc[0]:.4f}")
            st.metric("Median Score", f"{top_df['final_score'].median():.4f}")
        with col2:
            st.metric("Bottom Score", f"{top_df['final_score'].iloc[-1]:.4f}")
            honeypots = int(scored_df["group_h_honeypot"].sum())
            st.metric("Honeypots Detected", honeypots)

        st.subheader("💬 Reasoning")
        for _, row in top_df.iterrows():
            with st.expander(f"Rank {int(row['rank'])}: {row['candidate_id']} (score: {row['final_score']:.4f})"):
                st.write(row["reasoning"])

        submission = top_df[["candidate_id", "rank", "final_score", "reasoning"]].copy()
        submission = submission.rename(columns={"final_score": "score"})
        csv_data = submission.to_csv(index=False, encoding="utf-8")
        st.download_button(
            "📥 Download Submission CSV",
            data=csv_data,
            file_name="team_submission.csv",
            mime="text/csv",
        )

        try:
            os.unlink(tmp_path)
        except OSError:
            pass  
        st.session_state.pop("tmp_path", None)
        st.session_state.pop("tmp_upload_name", None)
