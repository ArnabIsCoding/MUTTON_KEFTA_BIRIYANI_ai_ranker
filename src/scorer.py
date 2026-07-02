
import numpy as np
import pandas as pd
from src.config import (
    CAREER_TITLE_WEIGHT, CAREER_DESC_WEIGHT,
    MULTIPLICATIVE_FLOORS, CAREER_EXPONENT,
    EDUCATION_ADDITIVE_BONUS, TFIDF_ADDITIVE_BONUS,
    QUANTITATIVE_ADDITIVE_BONUS,
)


def compute_final_scores(features_list: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(features_list)
    floors = MULTIPLICATIVE_FLOORS

    df["career_relevance"] = (
        CAREER_TITLE_WEIGHT * df["group_a_title"] +
        CAREER_DESC_WEIGHT * df["group_b_desc"]
    )

    multiplicative_core = (
        df["career_relevance"].clip(lower=0.01) ** CAREER_EXPONENT
      * df["group_c_skills"].clip(lower=floors["skills"])
      * df["group_e_experience"].clip(lower=floors["experience"])
      * df["group_f_location"].clip(lower=floors["location"])
      * df["group_g_behavioral"].clip(lower=floors["behavioral"])
      * df["group_d_gate"]
      * (1.0 - df["group_h_honeypot"])
    )

    additive_bonus = (
        EDUCATION_ADDITIVE_BONUS    * df["group_i_education"]
      + TFIDF_ADDITIVE_BONUS       * df["group_j_semantic"]
      + QUANTITATIVE_ADDITIVE_BONUS * df["group_k_quantitative"]
    )

    df["final_score"] = (multiplicative_core + additive_bonus).round(6)

    df = df.sort_values(
        by=["final_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


def select_top_n(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    top = df.head(n).copy()
    top["rank"] = range(1, len(top) + 1)

    prev_score = float("inf")
    scores = top["final_score"].tolist()
    for i in range(len(scores)):
        if scores[i] > prev_score:
            scores[i] = prev_score
        prev_score = scores[i]
    top["final_score"] = scores

    return top

