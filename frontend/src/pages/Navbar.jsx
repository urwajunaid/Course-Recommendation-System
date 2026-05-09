import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import styles from "./Navbar.module.css";

export default function Navbar() {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  return (
    <nav className={styles.nav}>
      <Link to="/" className={styles.logo}>
        <span className={styles.logoIcon}>◈</span>
        <span>CourseRec</span>
      </Link>

      <div className={styles.links}>
        <Link to="/"       className={location.pathname === "/"       ? styles.active : ""}>Home</Link>
        <Link to="/search" className={location.pathname === "/search" ? styles.active : ""}>Search</Link>
        {user && (
          <Link to="/profile" className={location.pathname === "/profile" ? styles.active : ""}>
            Profile
          </Link>
        )}
      </div>

      <div className={styles.actions}>
        {user ? (
          <>
            <span className={styles.userBadge}>{user.username}</span>
            <button className={styles.btnOutline} onClick={handleLogout}>Logout</button>
          </>
        ) : (
          <Link to="/auth" className={styles.btnPrimary}>Sign In</Link>
        )}
      </div>
    </nav>
  );
}
