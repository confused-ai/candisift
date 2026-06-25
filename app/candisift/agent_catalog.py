"""Static description of the agent roster — what each agent does and which model
tier it runs on. Joined with live tracer stats for the Agents UI/API."""
from __future__ import annotations


def agents(persona_model: str, synth_model: str) -> list[dict]:
    return [
        {"role": "profile", "name": "Profile Extractor", "stage": "ingest",
         "model_tier": persona_model, "tools": [],
         "does": "Structures raw resume text (skills, years, auth) into a profile."},
        {"role": "jd", "name": "JD Extractor", "stage": "job setup",
         "model_tier": persona_model, "tools": [],
         "does": "Turns a job description into a structured requirements spec."},
        {"role": "tech", "name": "Technical Evaluator", "stage": "screen",
         "model_tier": persona_model, "tools": ["recall_similar_candidates"],
         "does": "Maps skills to the spec with evidence; scores technical depth."},
        {"role": "risk", "name": "Risk Evaluator", "stage": "screen",
         "model_tier": persona_model, "tools": [],
         "does": "Flags gaps, concurrent full-time roles, inconsistent dates."},
        {"role": "hr", "name": "People / HR Evaluator", "stage": "screen",
         "model_tier": persona_model, "tools": [],
         "does": "Reads people-fit: communication, collaboration, trajectory, motivation, culture-add."},
        {"role": "synth", "name": "Synthesizer", "stage": "screen",
         "model_tier": synth_model, "tools": ["recall_recruiter_feedback"],
         "does": "Combines tech + risk into the final fit verdict and recommendation."},
        {"role": "tool", "name": "Memory / Retrieval", "stage": "screen",
         "model_tier": "—", "tools": ["recall_similar_candidates", "recall_recruiter_feedback"],
         "does": "Persistent agent memory: recalls past candidates and recruiter decisions."},
    ]
