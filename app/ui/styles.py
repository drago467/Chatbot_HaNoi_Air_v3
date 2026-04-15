"""Custom CSS for the HanoiAir Chatbot Streamlit UI.

Light, elegant theme with cool blue tones. Supports left sidebar,
main chat area, and collapsible right info panel.
"""

CUSTOM_CSS = """
<style>
/* ── Hide Streamlit defaults ──────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stHeader"] {
    background: transparent;
    backdrop-filter: none;
}
[data-testid="stHeader"]::after { display: none; }

/* ── Global ───────────────────────────────────────────────────── */
.stApp {
    background-color: #f7f8fa;
}

/* ── Left sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 15px;
}

/* ── Sidebar header ───────────────────────────────────────────── */
.sidebar-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 0 12px 0;
}
.sidebar-header-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.sidebar-header-text {
    font-size: 18px;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.2;
}
.sidebar-header-sub {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 400;
}

/* ── Conversation list ────────────────────────────────────────── */
.conv-group-label {
    font-size: 12px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 12px 0 4px 4px;
    margin: 0;
}

/* ── Weather card (compact) ───────────────────────────────────── */
.weather-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 14px;
    padding: 16px 18px;
    color: white;
    margin: 4px 0;
}
.weather-card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.weather-card-temp {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
}
.weather-card-emoji {
    font-size: 36px;
}
.weather-card-desc {
    font-size: 15px;
    opacity: 0.9;
    margin-bottom: 10px;
}
.weather-card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
}
.weather-card-item {
    font-size: 13px;
    opacity: 0.85;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ── Right info panel ─────────────────────────────────────────── */
.info-panel-header {
    font-size: 16px;
    font-weight: 600;
    color: #1e293b;
    margin: 12px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Welcome message ──────────────────────────────────────────── */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px 30px;
    text-align: center;
}
.welcome-icon {
    font-size: 60px;
    margin-bottom: 16px;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.welcome-title {
    font-size: 26px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 8px;
}
.welcome-subtitle {
    font-size: 16px;
    color: #64748b;
    max-width: 450px;
    line-height: 1.5;
    margin-bottom: 24px;
}

/* ── Chat area ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}

/* Chat text — larger font, proper spacing */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    font-size: 17px;
    line-height: 1.7;
    color: #1e293b;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.6em;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
    margin: 0.4em 0 0.4em 1.5em;
    padding: 0;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    margin-bottom: 0.3em;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
    margin-top: 0.8em;
    margin-bottom: 0.4em;
    font-weight: 600;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
    background: #f1f5f9;
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
    background: #1e293b;
    color: #e2e8f0;
    padding: 12px 16px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 0.6em 0;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code {
    background: none;
    padding: 0;
    color: inherit;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
    border-collapse: collapse;
    margin: 0.6em 0;
    font-size: 0.9em;
    width: 100%;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
    border: 1px solid #e2e8f0;
    padding: 6px 10px;
    text-align: left;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
    background: #f8fafc;
    font-weight: 600;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
    font-weight: 600;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
    color: #64748b;
    margin: 0.6em 0;
}

/* Assistant avatar */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {
    background-color: #3b82f6 !important;
}

/* Suggestion button styling */
div.stButton > button[kind="secondary"] {
    border-radius: 20px;
    border: 1px solid #d1d5db;
    background: white;
    color: #374151;
    font-size: 14px;
    padding: 8px 18px;
    transition: all 0.15s;
}
div.stButton > button[kind="secondary"]:hover {
    background: #eff6ff;
    border-color: #93c5fd;
    color: #2563eb;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 4px;
}
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 4px;
}
[data-testid="stSidebar"] ::-webkit-scrollbar-track {
    background: transparent;
}
</style>
"""
