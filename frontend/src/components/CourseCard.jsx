import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { api } from "../api";
import styles from "./CourseCard.module.css";

export default function CourseCard({ course, rank }) {
  const { user } = useAuth();
  const [rating, setRating]     = useState(0);
  const [hovered, setHovered]   = useState(0);
  const [saved, setSaved]       = useState(false);
  const [saving, setSaving]     = useState(false);
  const [rateError, setRateError] = useState("");
  const [expanded, setExpanded] = useState(false);

  const handleRate = async (stars) => {
    if (!user) return;
    setRating(stars);
    setSaving(true);
    setRateError("");
    try {
      await api.rateCourse(course.course_name, stars);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setRateError(err.message || "Failed to save rating");
      setTimeout(() => setRateError(""), 4000);
    } finally {
      setSaving(false);
    }
  };

  const diffColor = {
    "Beginner"     : "var(--success)",
    "Intermediate" : "var(--warn)",
    "Advanced"     : "var(--danger)",
    "Mixed"        : "var(--accent)",
  }[course.difficulty] || "var(--text-muted)";

  return (
    <div className={styles.card}>
      {/* Rank badge */}
      <div className={styles.rankBadge}>#{rank}</div>

      {/* Header */}
      <div className={styles.header}>
        <h3 className={styles.title}>{course.course_name}</h3>
        {course.company_name && (
          <span className={styles.company}>{course.company_name}</span>
        )}
      </div>

      {/* Meta pills */}
      <div className={styles.meta}>
        {course.difficulty && (
          <span className={styles.pill} style={{ color: diffColor, borderColor: diffColor + "44" }}>
            {course.difficulty}
          </span>
        )}
        {course.duration && (
          <span className={styles.pill}>⏱ {course.duration}</span>
        )}
        {course.certificate_type && (
          <span className={styles.pill}>🎓 {course.certificate_type}</span>
        )}
        {course.ratings != null && (
          <span className={styles.pill} style={{ color: "var(--warn)" }}>
            ★ {Number(course.ratings).toFixed(1)}
          </span>
        )}
      </div>

      {/* Score bar */}
      <div className={styles.scores}>
        {course.cbf_score != null && (
          <div className={styles.scoreRow}>
            <span className={styles.scoreLabel}>CBF</span>
            <div className={styles.scoreBar}>
              <div
                className={styles.scoreFill}
                style={{ width: `${Math.min(course.cbf_score * 100, 100)}%`, background: "var(--accent)" }}
              />
            </div>
            <span className={styles.scoreVal}>{course.cbf_score.toFixed(3)}</span>
          </div>
        )}
        {course.ncf_score != null && (
          <div className={styles.scoreRow}>
            <span className={styles.scoreLabel}>NCF</span>
            <div className={styles.scoreBar}>
              <div
                className={styles.scoreFill}
                style={{ width: `${((course.ncf_score - 1) / 4) * 100}%`, background: "var(--accent-2)" }}
              />
            </div>
            <span className={styles.scoreVal}>{course.ncf_score.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Skills (expandable) */}
      {course.skills && (
        <div className={styles.skillsWrap}>
          <button className={styles.skillToggle} onClick={() => setExpanded(!expanded)}>
            {expanded ? "▲ Hide skills" : "▼ Show skills"}
          </button>
          {expanded && <p className={styles.skills}>{course.skills}</p>}
        </div>
      )}

      {/* Star rating (only for logged-in users) */}
      {user && (
        <div className={styles.ratingRow}>
          <span className={styles.rateLabel}>
            {saving ? "Saving…" : saved ? "✓ Saved!" : "Rate:"}
          </span>
          <div className={styles.stars}>
            {[1, 2, 3, 4, 5].map((s) => (
              <button
                key={s}
                className={`${styles.star} ${(hovered || rating) >= s ? styles.starOn : ""}`}
                onMouseEnter={() => setHovered(s)}
                onMouseLeave={() => setHovered(0)}
                onClick={() => handleRate(s)}
                title={`Rate ${s}/5`}
              >
                ★
              </button>
            ))}
          </div>
          {rateError && (
            <span style={{ color: "var(--danger)", fontSize: "0.75rem", marginLeft: "0.5rem" }}>
              ⚠ {rateError}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
