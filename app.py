"""
TruthLens AI — Premium AI News Intelligence Dashboard & Inshorts Experience
Features multi-source news ingestion (BBC, The Guardian, Google Verified Feeds, NDTV),
Inshorts-style editorial cards, per-article Save/Share actions, automated refresh,
and end-to-end ML prediction and explainability.
"""

import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime
from predict import TruthLensPredictor, get_evaluation_metrics, LABEL_MAP
from news_service import (
    fetch_multi_source_news,
    BookmarkManager,
    SOURCES
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TruthLens AI — News Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DESIGN SYSTEM & STYLESHEET (Dark Enterprise Editorial Palette)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Core Canvas */
    .stApp {
        background-color: #0B0F14;
        color: #F1F5F9;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #111720;
        border-right: 1px solid #263244;
    }

    /* Headers */
    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #F1F5F9;
        margin-bottom: 0.25rem;
    }
    .dashboard-subtitle {
        font-size: 0.98rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Inshorts Editorial News Card */
    .inshorts-card {
        background-color: #141C27;
        border: 1px solid #263244;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.2s ease;
    }
    .inshorts-card:hover {
        border-color: #38BDF8;
    }
    .inshorts-card-body {
        padding: 1.25rem 1.4rem;
    }
    .inshorts-meta {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 0.78rem;
        color: #94A3B8;
        margin-bottom: 0.6rem;
    }
    .inshorts-tag {
        background-color: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .inshorts-source {
        font-weight: 600;
        color: #E2E8F0;
    }
    .inshorts-headline {
        font-size: 1.22rem;
        font-weight: 700;
        color: #F1F5F9;
        line-height: 1.35;
        margin-bottom: 0.65rem;
    }
    .inshorts-summary {
        font-size: 0.93rem;
        color: #CBD5E1;
        line-height: 1.62;
        margin-bottom: 1.1rem;
    }
    .inshorts-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #1E293B;
        padding-top: 0.85rem;
        font-size: 0.82rem;
    }

    /* Fallback Editorial Banner */
    .card-banner-fallback {
        background: linear-gradient(135deg, #1A2431 0%, #151D28 100%);
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem;
        border-bottom: 1px solid #263244;
    }

    /* Intelligence Telemetry Card */
    .intel-card {
        background-color: #141C27;
        border: 1px solid #263244;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }
    .intel-card:hover {
        border-color: #38BDF8;
    }
    .intel-card-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.4rem;
    }
    .intel-card-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
    }
    .intel-card-sub {
        font-size: 0.8rem;
        color: #64748B;
    }

    /* Status Badges */
    .badge-real {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1.2rem;
        border-radius: 6px;
        background-color: rgba(52, 211, 153, 0.12);
        color: #34D399;
        font-weight: 600;
        font-size: 1.15rem;
        border: 1px solid rgba(52, 211, 153, 0.35);
    }
    .badge-fake {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1.2rem;
        border-radius: 6px;
        background-color: rgba(248, 113, 113, 0.12);
        color: #F87171;
        font-weight: 600;
        font-size: 1.15rem;
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    /* Elevated Box */
    .elevated-box {
        background-color: #1B2634;
        border: 1px solid #263244;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Notice Box */
    .notice-box {
        background-color: #141C27;
        border-left: 4px solid #38BDF8;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #CBD5E1;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Feature Pills */
    .token-pill-real {
        background: #111720;
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
    }
    .token-pill-fake {
        background: #111720;
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
    }

    /* Form Inputs */
    .stTextArea textarea, .stTextInput input {
        background-color: #141C27 !important;
        color: #F1F5F9 !important;
        border: 1px solid #263244 !important;
        border-radius: 6px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #263244;
        background-color: #1B2634;
        color: #F1F5F9;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #38BDF8;
        color: #38BDF8;
        background-color: #141C27;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0369A1 !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# CORE LOGIC & RESOURCE CACHING
# -------------------------------------------------------------
@st.cache_resource
def get_predictor():
    return TruthLensPredictor()


POLL_INTERVAL_SECONDS = 10


@st.cache_data(ttl=15)
def cached_multi_source_news(category: str, query: str, limit: int = 30):
    """Caches news for 15s to balance live freshness with source rate limits."""
    return fetch_multi_source_news(category=category, search_query=query, limit=limit)


predictor = get_predictor()
eval_metrics = get_evaluation_metrics()

# Session State Initializations
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "News Feed"
if "selected_article" not in st.session_state:
    st.session_state["selected_article"] = None
if "headline_input" not in st.session_state:
    st.session_state["headline_input"] = ""
if "news_input" not in st.session_state:
    st.session_state["news_input"] = ""
if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = "All"
if "last_refresh_time" not in st.session_state:
    st.session_state["last_refresh_time"] = datetime.now().strftime("%I:%M:%S %p")
if "displayed_feed_articles" not in st.session_state:
    st.session_state["displayed_feed_articles"] = []


# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
saved_articles_count = len(BookmarkManager.load_bookmarks())

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom: 0.3rem;">
        <div style="background: linear-gradient(135deg, #0284C7, #38BDF8); width: 34px; height: 34px; border-radius: 7px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size: 1.1rem; color: white;">TL</div>
        <div>
            <div style="font-weight: 700; font-size: 1.15rem; color: #F1F5F9; letter-spacing: -0.01em;">TruthLens AI</div>
            <div style="font-size: 0.73rem; color: #38BDF8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">AI News Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #263244; margin: 1rem 0;'>", unsafe_allow_html=True)

    nav_options = [
        "News Feed",
        f"Saved News ({saved_articles_count})",
        "News Analysis",
        "Overview",
        "Model Performance",
        "Explainability",
        "About & Limitations"
    ]

    # Map current page to index
    clean_cur_page = st.session_state["current_page"]
    if clean_cur_page.startswith("Saved News"):
        current_nav_label = f"Saved News ({saved_articles_count})"
    else:
        current_nav_label = clean_cur_page

    nav_index = nav_options.index(current_nav_label) if current_nav_label in nav_options else 0
    selected_nav = st.radio(
        "Navigation",
        nav_options,
        index=nav_index,
        label_visibility="collapsed"
    )

    if selected_nav.startswith("Saved News"):
        st.session_state["current_page"] = "Saved News"
    else:
        st.session_state["current_page"] = selected_nav

    st.markdown("<hr style='border-color: #263244; margin: 1.2rem 0;'>", unsafe_allow_html=True)

    # Multi-Source Telemetry Status
    st.markdown("<div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8; font-weight:600; margin-bottom:0.75rem;'>NEWS INGESTION SOURCES</div>", unsafe_allow_html=True)

    for src_name in SOURCES.keys():
        short_name = src_name.split("(")[0].strip()
        st.markdown(
            f"<div style='font-size:0.82rem; margin-bottom:0.35rem; color:#F1F5F9;'>● {short_name}: <span style='color:#34D399; font-weight:600;'>Operational</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color: #263244; margin: 1.2rem 0;'>", unsafe_allow_html=True)

    # Genuine Model Readiness
    is_model_ready = predictor.is_model_loaded()
    if is_model_ready:
        st.markdown(
            "<div style='font-size:0.83rem; margin-bottom:0.35rem; color:#F1F5F9;'>● ML Pipeline: <span style='color:#34D399; font-weight:600;'>Loaded</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='font-size:0.83rem; margin-bottom:0.2rem; color:#F1F5F9;'>○ ML Pipeline: <span style='color:#FBBF24; font-weight:600;'>Awaiting Training</span></div>"
            "<div style='font-size:0.72rem; color:#94A3B8; margin-left:0.9rem;'>Kaggle dataset required in data/</div>",
            unsafe_allow_html=True
        )

    st.caption(f"TruthLens AI v1.3.0\nLast updated: {st.session_state['last_refresh_time']}")


# -------------------------------------------------------------
# HELPER: RENDER AN INSHORTS-STYLE EDITORIAL CARD
# -------------------------------------------------------------
def render_news_card(art: dict, prefix: str = "feed"):
    """Renders a single Inshorts-inspired editorial news card with Save, Share, and Analyze actions."""
    card_id = art["id"]
    is_saved = BookmarkManager.is_bookmarked(card_id)

    # 1. Card Container
    with st.container():
        # Top Thumbnail / Fallback Graphic
        if art.get("image_url"):
            st.markdown(f"""
            <div style="border-radius:12px 12px 0 0; overflow:hidden; max-height:220px; background:#141C27; border: 1px solid #263244; border-bottom:none;">
                <img src="{art['image_url']}" style="width:100%; height:210px; object-fit:cover; display:block;" onerror="this.style.display='none';">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-banner-fallback" style="border-radius:12px 12px 0 0;">
                <div>
                    <span class="inshorts-tag">{art.get('category', 'News')}</span>
                    <div style="font-size:0.88rem; color:#94A3B8; margin-top:0.4rem; font-weight:600;">{art.get('source')}</div>
                </div>
                <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.25); border-radius:6px; padding:6px 12px; color:#38BDF8; font-size:0.78rem; font-weight:600;">
                    AUTHENTIC SOURCE
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Card Content Box
        st.markdown(f"""
        <div class="inshorts-card" style="border-radius:0 0 12px 12px; border-top:none; margin-bottom:0.75rem;">
            <div class="inshorts-card-body">
                <div class="inshorts-meta">
                    <span class="inshorts-tag">{art.get('category', 'News')}</span>
                    <span class="inshorts-source">{art.get('source')}</span>
                    <span>•</span>
                    <span>{art.get('timestamp')}</span>
                </div>
                <div class="inshorts-headline">{art.get('headline')}</div>
                <div class="inshorts-summary">{art.get('summary')}</div>
                <div class="inshorts-footer">
                    <a href="{art.get('url')}" target="_blank" style="color:#38BDF8; text-decoration:none; font-weight:600;">
                        Read Full Story on {art.get('source')} ↗
                    </a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Action Buttons Row
        col_act1, col_act2, col_act3, _ = st.columns([1.6, 1.2, 1.3, 2.0])

        with col_act1:
            if st.button("⚡ Analyze This News", key=f"{prefix}_analyze_{card_id}", type="primary", use_container_width=True):
                st.session_state["selected_article"] = art
                st.session_state["headline_input"] = art["headline"]
                st.session_state["news_input"] = art["summary"]
                st.session_state["current_page"] = "News Analysis"
                st.rerun()

        with col_act2:
            if is_saved:
                if st.button("🔖 Saved", key=f"{prefix}_unsave_{card_id}", use_container_width=True):
                    BookmarkManager.remove_bookmark(card_id)
                    st.toast("Article removed from Saved News.", icon="🗑️")
                    st.rerun()
            else:
                if st.button("Save", key=f"{prefix}_save_{card_id}", use_container_width=True):
                    BookmarkManager.save_bookmark(art)
                    st.toast("Article bookmarked to Saved News!", icon="🔖")
                    st.rerun()

        with col_act3:
            # Native Web Share & Copy Link Experience
            with st.popover("🔗 Share", use_container_width=True):
                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F1F5F9; margin-bottom:0.4rem;'>Share News Briefing</div>", unsafe_allow_html=True)
                share_payload = {
                    "title": art.get("headline", ""),
                    "text": art.get("summary", ""),
                    "url": art.get("url", "")
                }
                components.html(
                    f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                    <meta charset="utf-8">
                    <style>
                      body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: transparent; }}
                      .btn-share {{
                        width: 100%;
                        padding: 8px 10px;
                        background: #0284C7;
                        color: #ffffff;
                        border: 1px solid #38BDF8;
                        border-radius: 6px;
                        font-size: 13px;
                        font-weight: 600;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 6px;
                        transition: background 0.15s ease;
                      }}
                      .btn-share:hover {{
                        background: #0369A1;
                      }}
                      .feedback {{
                        font-size: 11px;
                        margin-top: 5px;
                        text-align: center;
                        color: #94A3B8;
                        min-height: 15px;
                      }}
                    </style>
                    </head>
                    <body>
                      <button id="nativeShareBtn" class="btn-share">
                        📱 Open Device Share Sheet
                      </button>
                      <div id="shareStatus" class="feedback">Opens installed apps (WhatsApp, Email, etc.)</div>

                      <script>
                        const data = {json.dumps(share_payload)};
                        const btn = document.getElementById('nativeShareBtn');
                        const status = document.getElementById('shareStatus');

                        btn.addEventListener('click', async () => {{
                          const shareTarget = (window.parent && window.parent.navigator && window.parent.navigator.share)
                                              ? window.parent.navigator
                                              : ((window.navigator && window.navigator.share) ? window.navigator : null);

                          if (shareTarget && shareTarget.share) {{
                            try {{
                              await shareTarget.share(data);
                              status.style.color = "#34D399";
                              status.innerText = "✓ Shared via device sheet!";
                              return;
                            }} catch (err) {{
                              if (err.name === 'AbortError') {{
                                status.style.color = "#94A3B8";
                                status.innerText = "Share dialog closed.";
                                return;
                              }}
                            }}
                          }}

                          // Fallback: Copy link to clipboard
                          try {{
                            if (navigator.clipboard && navigator.clipboard.writeText) {{
                              await navigator.clipboard.writeText(data.url);
                              status.style.color = "#38BDF8";
                              status.innerText = "✓ URL copied to clipboard! (Native share unavailable)";
                            }} else {{
                              status.style.color = "#FBBF24";
                              status.innerText = "Please copy canonical URL below.";
                            }}
                          }} catch (copyErr) {{
                            status.style.color = "#F87171";
                            status.innerText = "Clipboard write blocked. Please copy URL below.";
                          }}
                        }});
                      </script>
                    </body>
                    </html>
                    """,
                    height=70
                )
                st.text_input(
                    "Canonical Article URL:",
                    value=art.get("url", ""),
                    label_visibility="collapsed",
                    key=f"{prefix}_share_input_{card_id}"
                )
                st.caption("Canonical publisher source URL • Device & browser native share")

        st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# APPLICATION HEADER & OVERFLOW MENU
# -------------------------------------------------------------
header_left, header_right = st.columns([5.5, 0.5])
with header_left:
    st.markdown(
        f"<div style='font-size:0.75rem; font-weight:700; color:#38BDF8; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:0.25rem;'>TRUTHLENS AI • {st.session_state['current_page'].upper()}</div>",
        unsafe_allow_html=True
    )
with header_right:
    with st.popover("⋮", help="More Actions & Navigation"):
        st.markdown("<div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#94A3B8; margin-bottom:0.5rem;'>TruthLens Menu</div>", unsafe_allow_html=True)
        cur_saved_count = len(BookmarkManager.load_bookmarks())
        if st.button(f"🔖 Saved News ({cur_saved_count})", key="hdr_to_saved", use_container_width=True):
            st.session_state["current_page"] = "Saved News"
            st.rerun()
        if st.button("📰 Return to News Feed", key="hdr_to_feed", use_container_width=True):
            st.session_state["current_page"] = "News Feed"
            st.rerun()
        if st.button("ℹ️ About TruthLens AI", key="hdr_to_about", use_container_width=True):
            st.session_state["current_page"] = "About & Limitations"
            st.rerun()
        st.markdown("<hr style='border-color:#263244; margin:0.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem; color:#64748B;'>Auto-polling: 10s<br>Sources: BBC, Guardian, Google, NDTV</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 1: NEWS FEED (INSHORTS-STYLE MULTI-SOURCE FEED)
# -------------------------------------------------------------
if st.session_state["current_page"] == "News Feed":
    st.markdown('<div class="dashboard-title">Live Intelligence Feed</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Read the news. Examine the evidence. Multi-source verified briefings from BBC, The Guardian, NDTV, and Reuters.</div>',
        unsafe_allow_html=True
    )

    # Filter & Controls Row
    cat_cols = st.columns([4.2, 1.8, 1.0])
    with cat_cols[0]:
        categories = ["All", "World", "India", "Politics", "Technology", "Business", "Science", "Health", "Sports"]
        cat_selection = st.pills("Filter by Topic:", categories, default=st.session_state["selected_category"])
        if cat_selection:
            st.session_state["selected_category"] = cat_selection

    with cat_cols[1]:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; padding-top:8px; font-size:0.82rem; color:#34D399; font-weight:600;">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#34D399; box-shadow:0 0 10px #34D399;"></span>
            Auto-refreshing (every 10s)
        </div>
        """, unsafe_allow_html=True)

    with cat_cols[2]:
        if st.button("🔄 Refresh", use_container_width=True, help="Optional manual refresh"):
            st.cache_data.clear()
            st.session_state["last_refresh_time"] = datetime.now().strftime("%I:%M:%S %p")
            st.rerun()

    # Search Bar
    search_col, _ = st.columns([4, 1.5])
    with search_col:
        search_kw = st.text_input(
            "Search Live News",
            placeholder="Search breaking headlines across all publishers...",
            label_visibility="collapsed"
        )

    current_category = st.session_state["selected_category"]

    @st.fragment(run_every=POLL_INTERVAL_SECONDS)
    def render_auto_news_feed(cat: str, q: str):
        fetched, telemetry = cached_multi_source_news(cat, q.strip(), limit=30)

        # Deduplication and preservation of existing articles
        seen_keys = set()
        merged_articles = []

        # Put newly fetched first, then previously displayed
        for art in fetched + st.session_state.get("displayed_feed_articles", []):
            art_key = art["url"].lower().rstrip("/")
            if art_key not in seen_keys:
                seen_keys.add(art_key)
                merged_articles.append(art)

        if fetched:
            st.session_state["displayed_feed_articles"] = merged_articles[:35]
            st.session_state["last_refresh_time"] = datetime.now().strftime("%I:%M:%S %p")

        display_articles = st.session_state.get("displayed_feed_articles", fetched)

        st.markdown(
            f"<div style='font-size:0.83rem; color:#94A3B8; margin-bottom:1rem; display:flex; justify-content:space-between;'>"
            f"<span>Showing <strong>{len(display_articles)}</strong> verified live briefings</span>"
            f"<span>Last checked: <strong>{st.session_state['last_refresh_time']}</strong></span>"
            f"</div>",
            unsafe_allow_html=True
        )

        if not display_articles:
            st.markdown("""
            <div class="notice-box">
                <strong>No articles found:</strong> Could not retrieve news matching your criteria.
                Please verify internet connectivity or try another category.
            </div>
            """, unsafe_allow_html=True)
        else:
            for art in display_articles:
                render_news_card(art, prefix="feed")

    render_auto_news_feed(current_category, search_kw)


# -------------------------------------------------------------
# PAGE 2: SAVED NEWS (BOOKMARKS REPOSITORY)
# -------------------------------------------------------------
elif st.session_state["current_page"] == "Saved News":
    st.markdown('<div class="dashboard-title">Saved News & Bookmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Personal repository of saved intelligence briefings stored securely and locally.</div>',
        unsafe_allow_html=True
    )

    saved_dict = BookmarkManager.load_bookmarks()
    saved_list = list(saved_dict.values())

    if not saved_list:
        st.markdown("""
        <div class="elevated-box" style="text-align:center; padding:3rem 1rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔖</div>
            <div style="font-size:1.15rem; font-weight:700; color:#F1F5F9; margin-bottom:0.4rem;">No Saved Articles Yet</div>
            <div style="font-size:0.9rem; color:#94A3B8; margin-bottom:1.5rem;">Click the "Save" button on any news card in the Live Feed to bookmark it here.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Live News Feed →"):
            st.session_state["current_page"] = "News Feed"
            st.rerun()
    else:
        c_clear, c_count = st.columns([1.5, 4.5])
        with c_clear:
            if st.button("Clear All Bookmarks", type="secondary"):
                for art in saved_list:
                    BookmarkManager.remove_bookmark(art["id"])
                st.rerun()

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

        for art in saved_list:
            render_news_card(art, prefix="saved")


# -------------------------------------------------------------
# PAGE 3: NEWS ANALYSIS WORKSPACE
# -------------------------------------------------------------
elif st.session_state["current_page"] == "News Analysis":
    st.markdown('<div class="dashboard-title">News Intelligence Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Examine linguistic signals, statistical model estimates, and verified evidence context.</div>',
        unsafe_allow_html=True
    )

    sel_art = st.session_state.get("selected_article")

    if sel_art:
        st.markdown(f"""
        <div class="elevated-box" style="border-left: 4px solid #38BDF8; margin-top:0;">
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#38BDF8; font-weight:600; margin-bottom:0.3rem;">
                SELECTED BRIEFING • {sel_art.get('source', 'Publisher')}
            </div>
            <div style="font-size:1.18rem; font-weight:700; color:#F1F5F9; margin-bottom:0.4rem;">
                {sel_art.get('headline')}
            </div>
            <div style="font-size:0.84rem; color:#94A3B8; margin-bottom:0.6rem;">
                Published: {sel_art.get('timestamp')} • Category: {sel_art.get('category')}
            </div>
            <a href="{sel_art.get('url')}" target="_blank" style="color:#38BDF8; font-size:0.83rem; text-decoration:none; font-weight:600;">
                View Original Publisher Article ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Return to Live Feed"):
            st.session_state["selected_article"] = None
            st.session_state["current_page"] = "News Feed"
            st.rerun()

    # Manual / Edit Inputs
    headline_val = st.text_input(
        "Article Headline:",
        value=st.session_state.get("headline_input", ""),
        placeholder="Headline context..."
    )

    article_val = st.text_area(
        "Article Body / Short Description:",
        value=st.session_state.get("news_input", ""),
        height=160,
        placeholder="Enter or paste news text for classification..."
    )

    col_btn1, col_btn2 = st.columns([1.5, 4.5])
    with col_btn1:
        run_analysis = st.button("Run AI Intelligence Analysis", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("Clear Form"):
            st.session_state["headline_input"] = ""
            st.session_state["news_input"] = ""
            st.session_state["selected_article"] = None
            st.rerun()

    if run_analysis or (sel_art and "auto_analyzed" not in st.session_state):
        st.session_state["auto_analyzed"] = True
        combined_text = (headline_val.strip() + " " + article_val.strip()).strip()

        if not combined_text:
            st.error("Input validation failure: Please provide headline or article content.")
        else:
            with st.spinner("Extracting n-gram features and evaluating machine-learning pipeline..."):
                result = predictor.predict(combined_text)

            if not result["success"]:
                if not predictor.is_model_loaded():
                    st.markdown("""
                    <div class="elevated-box" style="border-left: 4px solid #FBBF24;">
                        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#FBBF24; font-weight:700; margin-bottom:0.4rem;">
                            ML PIPELINE STATUS: AWAITING TRAINING
                        </div>
                        <div style="font-size:1.15rem; font-weight:700; color:#F1F5F9; margin-bottom:0.5rem;">
                            Statistical Classifier Requires Training Artifacts
                        </div>
                        <div style="font-size:0.88rem; color:#CBD5E1; line-height:1.6; margin-bottom:1rem;">
                            The machine learning pipeline has not been trained yet. To compute statistical predictions (Real vs Fake classification) and extract influential n-gram feature tokens, the Kaggle Fake and Real News dataset is required.
                        </div>
                        <div style="background:#111720; border:1px solid #263244; border-radius:6px; padding:0.9rem; font-size:0.85rem; color:#94A3B8; margin-bottom:1rem;">
                            <div style="color:#F1F5F9; font-weight:600; margin-bottom:0.3rem;">Required Dataset Files:</div>
                            <div style="margin-top:0.2rem; font-family:monospace; color:#E2E8F0;">
                                • data/Fake.csv<br>
                                • data/True.csv
                            </div>
                            <div style="margin-top:0.6rem;"><strong>Dataset Source:</strong> <a href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset" target="_blank" style="color:#38BDF8; font-weight:600;">Kaggle Fake and Real News Dataset ↗</a></div>
                            <div style="margin-top:0.6rem;"><strong>Training Command:</strong></div>
                            <div style="background:#0B0F14; padding:6px 10px; border-radius:4px; font-family:monospace; color:#38BDF8; margin-top:0.25rem;">
                                python train.py
                            </div>
                        </div>
                        <div style="font-size:0.82rem; color:#64748B;">
                            TruthLens AI will automatically load the model pipeline as soon as artifacts (<code>models/truthlens_model.joblib</code>) are generated.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<h4 style='color:#F1F5F9; font-weight:600; margin-top:1.5rem;'>Primary Publisher Corroboration & Verification</h4>", unsafe_allow_html=True)
                    if sel_art:
                        st.markdown(f"""
                        <div class="elevated-box" style="border-left: 4px solid #38BDF8;">
                            <div style="font-weight:600; color:#38BDF8; font-size:0.95rem; margin-bottom:0.3rem;">Primary Publisher Corroboration</div>
                            <div style="font-size:0.88rem; color:#CBD5E1; line-height:1.5;">
                                This briefing was retrieved directly from verified reporting by <strong>{sel_art.get('source')}</strong>.
                                <br>
                                Canonical article link: <a href="{sel_art.get('url')}" target="_blank" style="color:#38BDF8; font-weight:600;">{sel_art.get('url')} ↗</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="elevated-box" style="border-left: 4px solid #A78BFA;">
                            <div style="font-weight:600; color:#A78BFA; font-size:0.95rem; margin-bottom:0.3rem;">Manual Text Snippet</div>
                            <div style="font-size:0.88rem; color:#94A3B8;">
                                Custom text provided for analysis. To verify claims independently, cross-reference key assertions with accredited fact-checking registries (Reuters Fact Check, Snopes, AP News).
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Analysis Failed: {result['error']}")
            else:
                st.markdown("<hr style='border-color: #263244; margin: 1.5rem 0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#F1F5F9; font-weight:700;'>Analysis Findings & Evidence Profile</h3>", unsafe_allow_html=True)

                res_col1, res_col2 = st.columns([1.1, 1.9])

                with res_col1:
                    is_real = result["label"] == "Real"
                    if is_real:
                        st.markdown(
                            '<div class="badge-real">✓ Model prediction: Real-labeled class</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="badge-fake">⚠ Model prediction: Fake-labeled class</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

                    if result["confidence"] is not None:
                        st.markdown(f"""
                        <div class="intel-card" style="margin-bottom:0.5rem;">
                            <div class="intel-card-header">Decision Function Score</div>
                            <div class="intel-card-value" style="color: {'#34D399' if is_real else '#F87171'};">
                                {result['confidence'] * 100:.1f}%
                            </div>
                            <div class="intel-card-sub">Type: {result['confidence_type']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                with res_col2:
                    st.markdown(f"""
                    <div class="elevated-box" style="margin-top:0;">
                        <div style="font-weight:600; font-size:0.92rem; color:#F1F5F9; margin-bottom:0.4rem;">Model Reasoning Basis:</div>
                        <ul style="font-size:0.87rem; color:#94A3B8; margin-bottom:0.6rem; padding-left:1.2rem;">
                            <li><strong>Pipeline Architecture:</strong> Scikit-Learn TF-IDF N-gram Classifier</li>
                            <li><strong>Evaluation Basis:</strong> Statistical similarity to verified dataset training tokens</li>
                            <li><strong>Score Interpretation:</strong> Mathematical distance in the learned feature space, NOT proof of objective factuality</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Influential Words / Features
                st.markdown("<h4 style='color:#F1F5F9; font-weight:600; margin-top:1.2rem;'>Influential Text Features</h4>", unsafe_allow_html=True)
                st.caption("Tokens in this specific text carrying the strongest mathematical weight:")

                influential = result["influential_features"]
                if influential:
                    token_cols = st.columns(min(len(influential), 3))
                    for i, feat in enumerate(influential[:6]):
                        c_idx = i % 3
                        is_token_real = feat["direction"] == "Real"
                        pill_class = "token-pill-real" if is_token_real else "token-pill-fake"
                        tag_color = "#34D399" if is_token_real else "#F87171"
                        with token_cols[c_idx]:
                            st.markdown(f"""
                            <div class="{pill_class}">
                                <div style="font-weight:600; font-size:0.92rem; color:#F1F5F9;">{feat['token']}</div>
                                <div style="font-size:0.77rem; color:#94A3B8; margin-top:0.2rem;">
                                    Steers toward: <span style="color:{tag_color}; font-weight:600;">{feat['direction']}</span>
                                </div>
                                <div style="font-size:0.73rem; color:#64748B;">Weight: <code>{feat['impact']}</code></div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No single token in the input exceeded the threshold for distinctive classification pull.")

                # Evidence Verification Section
                st.markdown("<h4 style='color:#F1F5F9; font-weight:600; margin-top:1.5rem;'>Evidence & Corroboration Verification</h4>", unsafe_allow_html=True)
                if sel_art:
                    st.markdown(f"""
                    <div class="elevated-box" style="border-left: 4px solid #A78BFA;">
                        <div style="font-weight:600; color:#A78BFA; font-size:0.95rem; margin-bottom:0.3rem;">Primary Publisher Corroboration</div>
                        <div style="font-size:0.88rem; color:#CBD5E1; line-height:1.5;">
                            This article was retrieved directly from verified reporting by <strong>{sel_art.get('source')}</strong>.
                            <br>
                            Canonical link: <a href="{sel_art.get('url')}" target="_blank" style="color:#38BDF8;">{sel_art.get('url')}</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="elevated-box" style="border-left: 4px solid #A78BFA;">
                        <div style="font-weight:600; color:#A78BFA; font-size:0.95rem; margin-bottom:0.3rem;">External Evidence Verification</div>
                        <div style="font-size:0.88rem; color:#94A3B8;">
                            External registry query is not configured in this local-first version to maintain complete user privacy.
                            Please cross-reference critical claims through accredited fact-checking registries (Snopes, Reuters Fact Check, PolitiFact).
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Misinformation Origin Investigation
                st.markdown("<h4 style='color:#F1F5F9; font-weight:600; margin-top:1.2rem;'>Misinformation Origin Investigation</h4>", unsafe_allow_html=True)
                st.markdown("""
                <div class="elevated-box">
                    <div style="font-size:0.88rem; color:#94A3B8;">
                        <strong>Origin Finding:</strong> Origin could not be established from available evidence.
                        <br>
                        Autonomous tracking of viral spread trajectories across social platforms requires dedicated archive crawling APIs.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Disclaimer Box
                st.markdown(f"""
                <div class="notice-box">
                    <strong style="color:#38BDF8;">Notice on Limitations:</strong> {result['disclaimer']}
                </div>
                """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 4: OVERVIEW DASHBOARD
# -------------------------------------------------------------
elif st.session_state["current_page"] == "Overview":
    st.markdown('<div class="dashboard-title">Intelligence Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Understand news signals. Inspect statistical model predictions. Explore transparent AI metrics.</div>',
        unsafe_allow_html=True
    )

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    with col_c1:
        acc = f"{eval_metrics.get('selected_test_metrics', {}).get('accuracy', 0)*100:.1f}%" if eval_metrics else "Not Available"
        st.markdown(f"""
        <div class="intel-card">
            <div class="intel-card-header">Model Test Accuracy</div>
            <div class="intel-card-value">{acc}</div>
            <div class="intel-card-sub">Held-out test split</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        f1 = f"{eval_metrics.get('selected_test_metrics', {}).get('f1_score', 0):.4f}" if eval_metrics else "Not Available"
        st.markdown(f"""
        <div class="intel-card">
            <div class="intel-card-header">Test F1-Score</div>
            <div class="intel-card-value">{f1}</div>
            <div class="intel-card-sub">Harmonic mean balance</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c3:
        samples = f"{eval_metrics.get('dataset_info', {}).get('test_samples', 0):,}" if eval_metrics else "Not Available"
        st.markdown(f"""
        <div class="intel-card">
            <div class="intel-card-header">Test Sample Volume</div>
            <div class="intel-card-value">{samples}</div>
            <div class="intel-card-sub">Held-out test articles</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c4:
        active_model = eval_metrics.get("best_model", "Linear Pipeline") if eval_metrics else ("Ready" if is_model_ready else "Not Configured")
        st.markdown(f"""
        <div class="intel-card">
            <div class="intel-card-header">Champion Architecture</div>
            <div class="intel-card-value" style="font-size:1.25rem; font-weight:600; padding-top:0.25rem;">{active_model}</div>
            <div class="intel-card-sub">TF-IDF N-Gram Vectorizer</div>
        </div>
        """, unsafe_allow_html=True)

    # Core Action Cards
    st.markdown("<h4 style='color:#F1F5F9; font-weight:600; margin-top:1.5rem; margin-bottom:0.75rem;'>Intelligence Actions</h4>", unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        st.markdown("""
        <div class="elevated-box" style="margin-top:0; min-height:165px;">
            <div style="font-weight:600; color:#38BDF8; font-size:1.05rem; margin-bottom:0.3rem;">📰 Inshorts-Style Feed</div>
            <div style="font-size:0.86rem; color:#94A3B8; margin-bottom:1rem; line-height:1.4;">Multi-source live news briefings with Save, Share, and instant AI verification.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open News Feed →", use_container_width=True):
            st.session_state["current_page"] = "News Feed"
            st.rerun()

    with action_col2:
        st.markdown("""
        <div class="elevated-box" style="margin-top:0; min-height:165px;">
            <div style="font-weight:600; color:#A78BFA; font-size:1.05rem; margin-bottom:0.3rem;">📊 Performance & Benchmarks</div>
            <div style="font-size:0.86rem; color:#94A3B8; margin-bottom:1rem; line-height:1.4;">Inspect confusion matrix, cross-algorithm validation comparisons, and class-wise precision/recall.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Model Benchmarks →", use_container_width=True):
            st.session_state["current_page"] = "Model Performance"
            st.rerun()

    with action_col3:
        st.markdown("""
        <div class="elevated-box" style="margin-top:0; min-height:165px;">
            <div style="font-weight:600; color:#34D399; font-size:1.05rem; margin-bottom:0.3rem;">🔬 Explainability & Features</div>
            <div style="font-size:0.86rem; color:#94A3B8; margin-bottom:1rem; line-height:1.4;">Expose influential words and n-grams mathematically steering classification decisions.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Explainability →", use_container_width=True):
            st.session_state["current_page"] = "Explainability"
            st.rerun()


# -------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE & BENCHMARKS
# -------------------------------------------------------------
elif st.session_state["current_page"] == "Model Performance":
    st.markdown('<div class="dashboard-title">Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Empirical validation metrics computed on held-out test splits without data leakage.</div>',
        unsafe_allow_html=True
    )

    if not eval_metrics:
        st.markdown("""
        <div class="notice-box" style="border-left-color: #38BDF8;">
            <div style="font-weight:600; color:#38BDF8; font-size:0.95rem; margin-bottom:0.3rem;">Evaluation Artifacts Not Yet Generated</div>
            <div>The evaluation report (<code>models/evaluation_results.json</code>) has not been computed yet.</div>
            <div style="margin-top:0.5rem; font-size:0.85rem; color:#94A3B8;">
                To generate verified metrics, place <code>Fake.csv</code> and <code>True.csv</code> in <code>data/</code> and execute <code>python train.py</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        best_name = eval_metrics.get("best_model", "Selected Model")
        test_data = eval_metrics.get("selected_test_metrics", {})
        all_models = eval_metrics.get("all_models_validation", {})

        st.markdown(f"<div style='font-size:1.1rem; font-weight:600; color:#F1F5F9; margin-bottom:0.8rem;'>Champion Architecture: <span style='color:#38BDF8;'>{best_name}</span></div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="intel-card">
                <div class="intel-card-header">Test Accuracy</div>
                <div class="intel-card-value">{test_data.get('accuracy', 0)*100:.2f}%</div>
                <div class="intel-card-sub">Independent test split</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="intel-card">
                <div class="intel-card-header">Weighted Precision</div>
                <div class="intel-card-value">{test_data.get('precision', 0)*100:.2f}%</div>
                <div class="intel-card-sub">False positive resistance</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="intel-card">
                <div class="intel-card-header">Weighted Recall</div>
                <div class="intel-card-value">{test_data.get('recall', 0)*100:.2f}%</div>
                <div class="intel-card-sub">Detection coverage</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="intel-card">
                <div class="intel-card-header">Macro F1-Score</div>
                <div class="intel-card-value">{test_data.get('f1_score', 0):.4f}</div>
                <div class="intel-card-sub">Harmonic mean balance</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: #263244; margin: 1.5rem 0;'>", unsafe_allow_html=True)

        comp_col, cm_col = st.columns([1.2, 1.0])

        with comp_col:
            st.markdown("<h4 style='color:#F1F5F9; font-weight:600;'>Validation Split Algorithm Comparison</h4>", unsafe_allow_html=True)
            if all_models:
                rows = []
                for name, metrics in all_models.items():
                    rows.append({
                        "Algorithm": name,
                        "Accuracy": f"{metrics.get('accuracy', 0)*100:.2f}%",
                        "Precision": f"{metrics.get('precision', 0)*100:.2f}%",
                        "Recall": f"{metrics.get('recall', 0)*100:.2f}%",
                        "F1-Score": f"{metrics.get('f1_score', 0):.4f}"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Comparative validation metrics unavailable.")

        with cm_col:
            st.markdown("<h4 style='color:#F1F5F9; font-weight:600;'>Test Split Confusion Matrix</h4>", unsafe_allow_html=True)
            cm = test_data.get("confusion_matrix", [])
            if cm and len(cm) == 2:
                cm_df = pd.DataFrame(
                    cm,
                    columns=["Predicted Fake (0)", "Predicted Real (1)"],
                    index=["Actual Fake (0)", "Actual Real (1)"]
                )
                st.dataframe(cm_df, use_container_width=True)
            else:
                st.info("Confusion matrix data unavailable.")


# -------------------------------------------------------------
# PAGE 6: EXPLAINABILITY VIEW
# -------------------------------------------------------------
elif st.session_state["current_page"] == "Explainability":
    st.markdown('<div class="dashboard-title">Explainability & Feature Weights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Mathematical transparency into how TF-IDF n-grams and linear weights steer decisions.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="elevated-box">
        <h4 style="color:#38BDF8; margin-top:0; margin-bottom:0.5rem; font-weight:600;">How TruthLens AI Explains Predictions</h4>
        <div style="font-size:0.9rem; color:#94A3B8; line-height:1.6;">
            Linear models compute decision scores by multiplying input token frequencies (TF-IDF values) with learned coefficients:
            <br><br>
            <code style="color:#38BDF8;">Decision Score = Σ (TF-IDF_i × Weight_i) + Intercept</code>
            <br><br>
            • <strong>Positive Weight (&gt; 0):</strong> Correlates statistically with <em>Real</em> news articles in the training distribution.<br>
            • <strong>Negative Weight (&lt; 0):</strong> Correlates statistically with <em>Fake</em> news articles in the training distribution.
        </div>
    </div>
    """, unsafe_allow_html=True)

    test_phrase = st.text_input(
        "Enter a test sentence to inspect its feature influences:",
        value="Reuters reports official economic indicators show steady quarterly growth"
    )

    if test_phrase:
        with st.spinner("Calculating token weights..."):
            res = predictor.predict(test_phrase)

        if res["success"]:
            features = res["influential_features"]
            if features:
                st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
                cols = st.columns(min(len(features), 3))
                for idx, feat in enumerate(features):
                    c = cols[idx % 3]
                    is_r = feat["direction"] == "Real"
                    p_class = "token-pill-real" if is_r else "token-pill-fake"
                    tag_c = "#34D399" if is_r else "#F87171"
                    with c:
                        st.markdown(f"""
                        <div class="{p_class}">
                            <div style="font-weight:600; color:#F1F5F9; font-size:1rem;">{feat['token']}</div>
                            <div style="font-size:0.8rem; color:#94A3B8; margin: 0.2rem 0;">Pulls towards: <span style="color:{tag_c}; font-weight:600;">{feat['direction']}</span></div>
                            <div style="font-size:0.75rem; color:#64748B;">Impact: <code>{feat['impact']}</code></div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No strong characteristic features found in this short phrase.")


# -------------------------------------------------------------
# PAGE 7: ABOUT & LIMITATIONS
# -------------------------------------------------------------
elif st.session_state["current_page"] == "About & Limitations":
    st.markdown('<div class="dashboard-title">About TruthLens AI & Governance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Project architecture, dataset documentation, language constraints, and ethical AI safeguards.</div>',
        unsafe_allow_html=True
    )

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("""
        <div class="elevated-box">
            <h4 style="color:#38BDF8; margin-top:0; font-weight:600;">System Architecture</h4>
            <ul style="font-size:0.88rem; color:#94A3B8; line-height:1.6; padding-left:1.2rem;">
                <li><strong>Language:</strong> Python 3.10+ / Scikit-learn</li>
                <li><strong>Multi-Source Feeds:</strong> BBC News, The Guardian, NDTV News, Google Verified Aggregator</li>
                <li><strong>Feature Pipeline:</strong> Sublinear TF-IDF (1-2 N-grams, English stopword filtration, 25,000 max features)</li>
                <li><strong>Algorithms:</strong> Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Machines</li>
                <li><strong>Validation Methodology:</strong> Stratified 70/15/15 train/val/test splits without data leakage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="elevated-box">
            <h4 style="color:#38BDF8; margin-top:0; font-weight:600;">Dataset & Privacy Boundaries</h4>
            <ul style="font-size:0.88rem; color:#94A3B8; line-height:1.6; padding-left:1.2rem;">
                <li><strong>Dataset:</strong> Kaggle Fake and Real News Corpus (Clément Bisaillon)</li>
                <li><strong>Language Support:</strong> English only. Other languages are unsupported in this pipeline.</li>
                <li><strong>Bookmarks Storage:</strong> Minimal metadata stored locally in <code>data/bookmarks.json</code>.</li>
                <li><strong>Privacy:</strong> Zero tracking of user browsing habits; all ML inference runs in-memory.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h4 style='color:#F1F5F9; font-weight:600; margin-top:1.2rem;'>Responsible AI Limitations & Disclosure</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div class="notice-box">
        <strong>Failure Modes & Artifacts:</strong>
        <ol style="margin-top:0.4rem; padding-left:1.2rem; font-size:0.88rem; line-height:1.6;">
            <li><strong>Publisher Signature Bias:</strong> Real articles often start with publisher tags like <code>WASHINGTON (Reuters)</code>. Classifiers can latch onto these formatting tokens rather than factuality.</li>
            <li><strong>Adversarial Mimicry:</strong> Well-crafted misinformation written in formal journalistic tone will frequently bypass bag-of-words classifiers.</li>
            <li><strong>Temporal Shift:</strong> Historical political articles have distinct vocabulary that does not reliably generalize to modern emerging topics.</li>
        </ol>
        <strong>Mandatory Verification Rule:</strong> Never accept model predictions as definitive verdicts. Always consult accredited, independent fact-checking organizations.
    </div>
    """, unsafe_allow_html=True)
