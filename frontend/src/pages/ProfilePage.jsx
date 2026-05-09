import { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { api } from "../api";
import styles from "./ProfilePage.module.css";

export default function ProfilePage() {
  const { user, logout } = useAuth();
  const [ratings, setRatings] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.myRatings()
      .then(setRatings)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const stars = (n) => "★".repeat(Math.round(n)) + "☆".repeat(5 - Math.round(n));

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        {/* Profile header */}
        <div className={styles.header}>
          <div className={styles.avatar}>
            {user.username.slice(0, 2).toUpperCase()}
          </div>
          <div>
            <h1 className={styles.name}>{user.username}</h1>
            <p className={styles.email}>{user.email}</p>
            <div className={styles.idBadge}>NCF ID: {user.ncf_user_id}</div>
          </div>
          <button className={styles.logoutBtn} onClick={logout}>Logout</button>
        </div>

        {/* Stats */}
        <div className={styles.statsRow}>
          <div className={styles.stat}>
            <span className={styles.statNum}>{ratings.length}</span>
            <span className={styles.statLabel}>Courses Rated</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statNum}>
              {ratings.length > 0
                ? (ratings.reduce((a, r) => a + r.rating, 0) / ratings.length).toFixed(1)
                : "—"}
            </span>
            <span className={styles.statLabel}>Avg Rating</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statNum}>{user.ncf_user_id}</span>
            <span className={styles.statLabel}>Model ID</span>
          </div>
        </div>

        {/* Ratings history */}
        <h2 className={styles.sectionTitle}>Your Rated Courses</h2>
        {loading ? (
          <p className={styles.muted}>Loading…</p>
        ) : ratings.length === 0 ? (
          <div className={styles.emptyRatings}>
            <p>You haven't rated any courses yet.</p>
            <p>Find a course and rate it — the model will personalise your feed.</p>
            <a href="/search" className={styles.searchLink}>Browse Courses →</a>
          </div>
        ) : (
          <div className={styles.ratingsList}>
            {ratings.map((r, i) => (
              <div key={i} className={styles.ratingItem}>
                <div className={styles.ratingCourse}>{r.course_name}</div>
                <div className={styles.ratingStars} style={{ color: "var(--warn)" }}>
                  {stars(r.rating)}
                </div>
                <div className={styles.ratingDate}>
                  {new Date(r.rated_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
