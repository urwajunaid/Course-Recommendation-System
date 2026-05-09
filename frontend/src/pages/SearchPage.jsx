import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";
import CourseCard from "../components/CourseCard";
import styles from "./SearchPage.module.css";

export default function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { user } = useAuth();

  const [query, setQuery]       = useState(searchParams.get("q") || "");
  const [results, setResults]   = useState(null);
  const [meta, setMeta]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");
  const [topN, setTopN]         = useState(5);
  const [suggestions, setSuggestions] = useState([]);

  // Run search on mount if URL has ?q=
  useEffect(() => {
    const q = searchParams.get("q");
    if (q) runSearch(q);
  }, []);

  // Autocomplete
  useEffect(() => {
    if (query.length < 2) { setSuggestions([]); return; }
    const t = setTimeout(() => {
      api.autocomplete(query).then(setSuggestions).catch(() => setSuggestions([]));
    }, 250);
    return () => clearTimeout(t);
  }, [query]);

  const runSearch = async (q, n = topN) => {
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    setResults(null);
    setSuggestions([]);
    try {
      const data = await api.recommend(q.trim(), n);
      setResults(data.results);
      setMeta({ mode: data.mode, query: data.query_course, user: data.user_id });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setSearchParams({ q: query });
    runSearch(query);
  };

  const handleSuggestion = (name) => {
    setQuery(name);
    setSuggestions([]);
    setSearchParams({ q: name });
    runSearch(name);
  };

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <h1 className={styles.title}>Find Similar Courses</h1>
        <p className={styles.sub}>
          Enter a course name to get {user ? "personalised hybrid" : "content-based"} recommendations.
        </p>

        <form className={styles.searchWrap} onSubmit={handleSubmit}>
          <div className={styles.searchRow}>
            <div className={styles.inputWrap}>
              <input
                className={styles.input}
                placeholder="e.g. google cybersecurity"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                autoComplete="off"
              />
              {suggestions.length > 0 && (
                <ul className={styles.suggestions}>
                  {suggestions.map((s) => (
                    <li key={s} onClick={() => handleSuggestion(s)}>
                      <span>→</span> {s}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <select
              className={styles.select}
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
            >
              {[3, 5, 8, 10, 15, 20].map((n) => (
                <option key={n} value={n}>Top {n}</option>
              ))}
            </select>

            <button className={styles.btn} type="submit" disabled={loading}>
              {loading ? "…" : "Recommend"}
            </button>
          </div>
        </form>

        {/* Mode badge */}
        {meta && (
          <div className={styles.metaRow}>
            <span className={`${styles.modeBadge} ${meta.mode === "hybrid" ? styles.hybrid : styles.cbf}`}>
              {meta.mode === "hybrid" ? "⚡ Hybrid (CBF + NCF)" : "◎ Content-Based"}
            </span>
            {meta.user && <span className={styles.userTag}>User: {meta.user}</span>}
            {!user && (
              <span className={styles.hint}>
                <a href="/auth">Sign in</a> for NCF-powered personalisation
              </span>
            )}
          </div>
        )}

        {/* Error */}
        {error && <div className={styles.error}>{error}</div>}

        {/* Loading */}
        {loading && (
          <div className={styles.grid}>
            {[...Array(topN)].map((_, i) => (
              <div key={i} className={styles.skeleton} />
            ))}
          </div>
        )}

        {/* Results */}
        {results && !loading && (
          results.length === 0 ? (
            <p className={styles.empty}>No results found. Try a different course name.</p>
          ) : (
            <div className={styles.grid}>
              {results.map((c, i) => (
                <CourseCard key={i} course={c} rank={c.rank || i + 1} />
              ))}
            </div>
          )
        )}

        {/* Empty state */}
        {!results && !loading && !error && (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>◈</div>
            <p>Type a course name above to get recommendations</p>
          </div>
        )}
      </div>
    </div>
  );
}
