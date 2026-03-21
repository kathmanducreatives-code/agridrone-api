import logging
import os
from typing import Any, Optional

logger = logging.getLogger("agridrone.report")


def call_openai_report(summary: dict[str, Any], mission_meta: dict[str, Any], sensor_stats: Optional[dict[str, Any]] = None):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    logger.info(
        "report.openai_stub_skipped",
        extra={"mission_id": mission_meta.get("missionId"), "reason": "stub_not_implemented"},
    )
    return None


def _severity_line(summary: dict[str, Any]) -> str:
    severity_distribution = summary.get("severity_distribution", {})
    if not severity_distribution:
        return "No disease detections were found in the uploaded imagery."
    parts = [f"{label}: {count}" for label, count in sorted(severity_distribution.items()) if count]
    return "Severity profile across the mission was " + ", ".join(parts) + "."


def _build_recommendations(summary: dict[str, Any]) -> list[str]:
    counts = summary.get("counts_per_disease", {})
    if not counts:
        return [
            "Continue routine scouting on the same route to confirm field health over time.",
            "Review image quality for any blurred captures before the next mission.",
        ]

    top_disease = max(counts, key=counts.get)
    recommendations = [
        f"Inspect the zones associated with {top_disease} first and compare with ground truth before treatment.",
        "Prioritize follow-up on the most severe detections captured in this mission report.",
    ]
    if summary.get("failed_images", 0):
        recommendations.append("Revisit failed or missing uploads to avoid blind spots in the final field assessment.")
    return recommendations


def generate_report(summary: dict[str, Any], mission_meta: dict[str, Any], sensor_stats: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    openai_report = call_openai_report(summary, mission_meta, sensor_stats=sensor_stats)
    if openai_report:
        return openai_report

    mission_id = mission_meta.get("missionId", "unknown")
    crop = mission_meta.get("crop", "rice")
    processed = int(summary.get("processed_images", 0))
    detections = int(summary.get("images_with_detections", 0))
    failed = int(summary.get("failed_images", 0))
    top_findings = summary.get("top_severe_detections", [])

    findings = []
    if processed:
        findings.append(f"Processed {processed} mission images for {crop}.")
    if detections:
        findings.append(f"{detections} images contained at least one disease detection.")
    else:
        findings.append("No disease detections were produced by the current model.")
    if failed:
        findings.append(f"{failed} images could not be downloaded or decoded and were excluded from analysis.")
    findings.append(_severity_line(summary))

    breakdown = []
    for disease, count in sorted(summary.get("counts_per_disease", {}).items()):
        breakdown.append(
            {
                "disease": disease,
                "count": count,
                "avg_confidence": round(summary.get("average_confidence_per_disease", {}).get(disease, 0.0), 4),
            }
        )

    severity_assessment = "low"
    severe_count = summary.get("severity_distribution", {}).get("severe", 0)
    moderate_count = summary.get("severity_distribution", {}).get("moderate", 0)
    if severe_count > 0:
        severity_assessment = "high"
    elif moderate_count > 0:
        severity_assessment = "medium"

    next_actions = [
        "Validate the highest-confidence detections with a targeted field walk.",
        "Use the mission summary for treatment planning rather than acting on single frames alone.",
    ]
    if top_findings:
        first = top_findings[0]
        next_actions.append(
            f"Review image {first.get('imageId')} first because it has the strongest severity signal in this batch."
        )

    return {
        "mission_id": mission_id,
        "crop": crop,
        "executive_summary": (
            f"Mission {mission_id} for {crop} completed with {processed} analyzed images, "
            f"{detections} images containing detections, and overall severity assessed as {severity_assessment}."
        ),
        "findings": findings,
        "disease_breakdown": breakdown,
        "severity_assessment": severity_assessment,
        "recommendations": _build_recommendations(summary),
        "next_actions": next_actions,
        "confidence_notes": (
            "Confidence scores come from the current onboard disease model and should be combined "
            "with field verification, especially where image blur, motion, or low resolution were observed."
        ),
        "top_severe_detections": top_findings,
        "sensor_stats": sensor_stats or {},
    }
