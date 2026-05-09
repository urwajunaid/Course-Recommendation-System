import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { api } from "../api";
import CourseCard from "../components/CourseCard";
import styles from "./HomePage.module.css";

export default function HomePage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [query, setQuery]         = useState("");
  const [feed, setFeed]           = useState([]);
  const [feedLoading, setFeedLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);

  // Load personalized feed if logged in
  useEffect(() => {
    if (!user) return;
    setFeedLoading(true);
    api.personalized(10)
      .then((data) => setFeed(data.results || []))
      .catch(console.error)
      .finally(() => setFeedLoading(false));
  }, [user]);

  // Autocomplete
  useEffect(() => {
    if (query.length < 2) { setSuggestions([]); return; }
    const t = setTimeout(() => {
      api.autocomplete(query).then(setSuggestions).catch(() => setSuggestions([]));
    }, 250);
    return () => clearTimeout(t);
  }, [query]);

  const handleSearch = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    navigate(`/search?q=${encodeURIComponent(query.trim())}`);
  };

  const handleSuggestion = (name) => {
    setQuery(name);
    setSuggestions([]);
    navigate(`/search?q=${encodeURIComponent(name)}`);
  };

  return (
    <div className={styles.page}>
      {/* Hero */}
      <section className={styles.hero}>
        <div className={styles.heroBg} />
        <div className={styles.heroContent}>
          <div className={styles.badge}>Hybrid AI Recommender</div>
          <h1 className={styles.heroTitle}>
            Discover courses<br />
            <span className={styles.heroAccent}>built for you</span>
          </h1>
          <p className={styles.heroSub}>
            CBF + NCF hybrid model surfaces the most relevant courses
            based on content similarity and your learning history.
          </p>

          <form className={styles.searchWrap} onSubmit={handleSearch}>
            <div className={styles.searchBox}>
              <span className={styles.searchIcon}>⌕</span>
              <input
                className={styles.searchInput}
                placeholder="Search a course, e.g. machine learning…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                autoComplete="off"
              />
              <button className={styles.searchBtn} type="submit">Search</button>
            </div>
            {suggestions.length > 0 && (
              <ul className={styles.suggestions}>
                {suggestions.map((s) => (
                  <li key={s} onClick={() => handleSuggestion(s)}>
                    <span className={styles.sugIcon}>→</span> {s}
                  </li>
                ))}
              </ul>
            )}
          </form>
        </div>
      </section>

      {/* Personalized feed */}
      {user && (
        <section className={styles.section}>
          <div className={styles.sectionHead}>
            <h2>For you, {user.username}</h2>
            <span className={styles.tag}>Personalised</span>
          </div>
          {feedLoading ? (
            <div className={styles.loadingRow}>
              {[...Array(4)].map((_, i) => <div key={i} className={styles.skeleton} />)}
            </div>
          ) : feed.length > 0 ? (
            <div className={styles.grid}>
              {feed.map((c, i) => <CourseCard key={i} course={c} rank={i + 1} />)}
            </div>
          ) : (
            <p className={styles.empty}>
              Rate a few courses to unlock personalised recommendations.
            </p>
          )}
        </section>
      )}

      {/* Guest CTA */}
      {!user && (
        <section className={styles.ctaSection}>
          <div className={styles.ctaCard}>
            <h2>Get personalised recommendations</h2>
            <p>Sign in and the NCF model will re-rank courses based on your taste.</p>
            <a href="/auth" className={styles.ctaBtn}>Sign In / Register</a>
          </div>
        </section>
      )}
    </div>
  );
}
