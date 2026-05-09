const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

function getToken() {
  return localStorage.getItem("token");
}

async function request(path, options = {}) {
  const token = getToken();
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  const res = await fetch(`${BASE}${path}`, { ...options, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || "Request failed");
  }

  return res.json();
}

export const api = {
  // Auth
  register: (data) =>
    request("/auth/register", { method: "POST", body: JSON.stringify(data) }),
  login: (data) =>
    request("/auth/login", { method: "POST", body: JSON.stringify(data) }),
  me: () => request("/auth/me"),

  // Courses
  searchCourses: (q) => request(`/courses/search?q=${encodeURIComponent(q)}`),
  autocomplete: (q) => request(`/courses/autocomplete?q=${encodeURIComponent(q)}`),
  rateCourse: (course_name, rating) =>
    request("/courses/rate", {
      method: "POST",
      body: JSON.stringify({ course_name, rating }),
    }),
  myRatings: () => request("/courses/my-ratings"),

  // Recommendations
  recommend: (course, top_n = 5) =>
    request(`/recommend?course=${encodeURIComponent(course)}&top_n=${top_n}`),
  personalized: (top_n = 10) =>
    request(`/recommend/personalized?top_n=${top_n}`),
};
