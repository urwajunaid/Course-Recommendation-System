import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import styles from "./AuthPage.module.css";

export default function AuthPage() {
  const { login, register } = useAuth();
  const navigate = useNavigate();

  const [tab, setTab]         = useState("login");   // "login" | "register"
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  // Login fields
  const [loginEmail, setLoginEmail]       = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  // Register fields
  const [regUsername, setRegUsername] = useState("");
  const [regEmail, setRegEmail]       = useState("");
  const [regPassword, setRegPassword] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(loginEmail, loginPassword);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    if (regPassword.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }
    setLoading(true);
    try {
      await register(regUsername, regEmail, regPassword);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        {/* Logo */}
        <div className={styles.logo}>
          <span className={styles.logoIcon}>◈</span>
          <span className={styles.logoText}>CourseRec</span>
        </div>

        <p className={styles.tagline}>
          AI-powered hybrid course recommendations
        </p>

        {/* Tabs */}
        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${tab === "login" ? styles.activeTab : ""}`}
            onClick={() => { setTab("login"); setError(""); }}
          >
            Sign In
          </button>
          <button
            className={`${styles.tab} ${tab === "register" ? styles.activeTab : ""}`}
            onClick={() => { setTab("register"); setError(""); }}
          >
            Register
          </button>
        </div>

        {/* Error */}
        {error && <div className={styles.error}>{error}</div>}

        {/* Login Form */}
        {tab === "login" && (
          <form className={styles.form} onSubmit={handleLogin}>
            <label className={styles.label}>Email</label>
            <input
              className={styles.input}
              type="email"
              placeholder="you@example.com"
              value={loginEmail}
              onChange={(e) => setLoginEmail(e.target.value)}
              required
            />
            <label className={styles.label}>Password</label>
            <input
              className={styles.input}
              type="password"
              placeholder="••••••••"
              value={loginPassword}
              onChange={(e) => setLoginPassword(e.target.value)}
              required
            />
            <button className={styles.btn} type="submit" disabled={loading}>
              {loading ? "Signing in…" : "Sign In"}
            </button>
          </form>
        )}

        {/* Register Form */}
        {tab === "register" && (
          <form className={styles.form} onSubmit={handleRegister}>
            <label className={styles.label}>Username</label>
            <input
              className={styles.input}
              type="text"
              placeholder="your_username"
              value={regUsername}
              onChange={(e) => setRegUsername(e.target.value)}
              required
            />
            <label className={styles.label}>Email</label>
            <input
              className={styles.input}
              type="email"
              placeholder="you@example.com"
              value={regEmail}
              onChange={(e) => setRegEmail(e.target.value)}
              required
            />
            <label className={styles.label}>Password</label>
            <input
              className={styles.input}
              type="password"
              placeholder="Min 6 characters"
              value={regPassword}
              onChange={(e) => setRegPassword(e.target.value)}
              required
            />
            <button className={styles.btn} type="submit" disabled={loading}>
              {loading ? "Creating account…" : "Create Account"}
            </button>
          </form>
        )}

        <p className={styles.footer}>
          {tab === "login" ? (
            <>No account? <button className={styles.switchLink} onClick={() => { setTab("register"); setError(""); }}>Register</button></>
          ) : (
            <>Have an account? <button className={styles.switchLink} onClick={() => { setTab("login"); setError(""); }}>Sign In</button></>
          )}
        </p>
      </div>
    </div>
  );
}
