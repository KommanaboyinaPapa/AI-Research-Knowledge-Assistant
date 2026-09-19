import { useEffect, useState, type FormEvent } from "react";
import { FileText, Send, Upload, Activity } from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
type DocumentItem = { id: string; filename: string; chunks: number };
type Source = { filename: string; text: string; distance: number; chunk_id?: string; chunk_number?: number; page?: number };
type Evaluation = { hit_rate: number; average_latency_ms: number; results?: { question: string; hit: boolean; latency_ms: number }[] };

export default function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [busy, setBusy] = useState(false);
  const [streamError, setStreamError] = useState("");
  const [token, setToken] = useState(() => localStorage.getItem("rag_token") ?? "");
  const [userEmail, setUserEmail] = useState("");
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");

  const authHeaders = { Authorization: `Bearer ${token}` };
  const loadDocuments = () => fetch(`${API}/api/documents`, { headers: authHeaders }).then((response) => response.ok ? response.json() : Promise.reject()).then(setDocuments).catch(() => setDocuments([]));
  useEffect(() => { if (token) { fetch(`${API}/api/auth/me`, { headers: authHeaders }).then((response) => response.ok ? response.json() : Promise.reject()).then((user) => { setUserEmail(user.email); loadDocuments(); }).catch(() => { localStorage.removeItem("rag_token"); setToken(""); }); } }, [token]);

  async function authenticate(event: FormEvent) {
    event.preventDefault(); setAuthError("");
    try {
      const path = authMode === "login" ? "login" : "register";
      const response = await fetch(`${API}/api/auth/${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.message ?? "Authentication failed.");
      if (authMode === "register") { setAuthMode("login"); setPassword(""); return; }
      localStorage.setItem("rag_token", data.access_token); setToken(data.access_token); setUserEmail(data.user.email); setPassword("");
    } catch (error) { setAuthError(error instanceof Error ? error.message : "Authentication failed."); }
  }

  function logout() { localStorage.removeItem("rag_token"); setToken(""); setUserEmail(""); setDocuments([]); }

  async function upload(file: File) {
    try {
      const body = new FormData(); body.append("file", file);
      const response = await fetch(`${API}/api/documents/upload`, { method: "POST", headers: authHeaders, body });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.message ?? "The document could not be uploaded.");
      }
      loadDocuments();
    } catch (error) { setStreamError(error instanceof Error ? error.message : "The document upload failed."); }
  }

  async function ask() {
    if (!question.trim()) return;
    setBusy(true);
    setStreamError(""); setAnswer(""); setSources([]);
    try {
      const response = await fetch(`${API}/api/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders }, body: JSON.stringify({ question }) });
      if (!response.ok || !response.body) throw new Error("The chat stream could not be started.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const lines = pending.split("\n"); pending = lines.pop() ?? "";
        for (const line of lines.filter(Boolean)) {
          const event = JSON.parse(line);
          if (event.type === "text") setAnswer((current) => current + event.text);
          if (event.type === "final") setSources(event.citations ?? []);
          if (event.type === "error") throw new Error(event.message);
        }
        if (done) break;
      }
    } catch (error) { setStreamError(error instanceof Error ? error.message : "The chat stream failed."); }
    finally { setBusy(false); }
  }

  async function runEvaluation() { setEvaluation(await fetch(`${API}/api/evaluation`, { headers: authHeaders }).then((response) => response.json())); }

  if (!token) return <main className="shell auth-shell"><header><div className="eyebrow"><Activity size={16} /> RESEARCH DESK</div><h1>Your research,<br /><em>in context.</em></h1><p className="lede">Sign in to keep your documents and conversations private.</p></header><form className="auth-form" onSubmit={authenticate}><label>Email<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label><label>Password<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} minLength={8} required /></label>{authError && <p className="stream-error">{authError}</p>}<button className="text-button" type="submit">{authMode === "login" ? "Log in" : "Create account"}</button><button className="text-button" type="button" onClick={() => { setAuthMode(authMode === "login" ? "register" : "login"); setAuthError(""); }}>{authMode === "login" ? "Register" : "Back to login"}</button></form></main>;

  return <main className="shell">
    <header><div className="eyebrow"><Activity size={16} /> RESEARCH DESK <button className="text-button" onClick={logout}>Log out {userEmail}</button></div><h1>Ask your documents<br /><em>better questions.</em></h1><p className="lede">A focused workspace for finding, comparing, and citing evidence across your research library.</p></header>
    <section className="workspace">
      <aside className="panel library"><div className="panel-title"><span>Library</span><label className="icon-button" title="Upload document"><Upload size={17} /><input type="file" accept=".pdf,.docx,.txt" onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])} /></label></div>{documents.length === 0 ? <p className="muted">No documents yet.</p> : documents.map((document) => <div className="document" key={document.id}><FileText size={18} /><span>{document.filename}<small>{document.chunks} chunks</small></span></div>)}<button className="text-button" onClick={runEvaluation}>Run evaluation</button>{evaluation && <><div className="metrics"><b>{Math.round(evaluation.hit_rate * 100)}%</b> hit rate <b>{evaluation.average_latency_ms}ms</b> avg latency</div><div className="evaluation-results">{evaluation.results?.map((item) => <div className="evaluation-row" key={item.question}><span>{item.hit ? "PASS" : "MISS"} · {item.question}</span><small>{item.latency_ms}ms</small></div>)}</div></>}</aside>
      <section className="conversation"><div className="answer"><span className="label">GROUNDED ANSWER</span>{busy && !answer ? <p className="muted">Generating...</p> : answer ? <p>{answer}</p> : <p className="muted">Upload a document, then ask a question to begin.</p>}{streamError && <p className="stream-error">{streamError}</p>}{sources.length > 0 && <div className="sources"><span className="label">SOURCES</span>{sources.map((source, index) => <details key={`${source.filename}-${index}`}><summary>{source.filename} · {source.page ? `page ${source.page} · ` : ""}chunk {source.chunk_number ?? index + 1} · distance {source.distance?.toFixed?.(2) ?? "-"}</summary><p>{source.text}</p></details>)}</div>}</div><form onSubmit={(event) => { event.preventDefault(); ask(); }}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What would you like to understand?" /><button className="send" disabled={busy} title="Ask question"><Send size={18} /></button></form></section>
    </section>
  </main>;
}