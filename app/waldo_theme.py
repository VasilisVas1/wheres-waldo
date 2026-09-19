"""Look and feel for the Streamlit demo: a "Where's Waldo?" storybook page.

Cream paper, Waldo red and white stripes, chunky lettering, hard offset shadows, pill buttons and a little
cartoon Waldo (drawn here as SVG — no image files to ship). Purely presentational: nothing in here touches the
models. `.streamlit/config.toml` sets the widget accent colours; the CSS below does the rest.
"""

from __future__ import annotations

import base64

RED = "#D62828"
INK = "#1F1A17"
PAPER = "#FBF3E4"
SKIN = "#F5CDA1"
JEANS = "#2B59B5"


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


_STRIPES = f"""
  <pattern id="stripes" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="#fff"/><rect width="20" height="10" fill="{RED}"/>
  </pattern>"""


def _head(cx: float = 70) -> str:
    """Hat, face and glasses, centred on x = cx (hat top ~y 18, chin ~y 112)."""
    dx = cx - 70
    return f"""
  <g transform="translate({dx} 0)">
    <path d="M40 70 C40 26 100 26 100 70 Z" fill="url(#stripes)" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>
    <rect x="35" y="64" width="70" height="13" rx="6.5" fill="{RED}" stroke="{INK}" stroke-width="3"/>
    <circle cx="70" cy="26" r="10" fill="#fff" stroke="{INK}" stroke-width="3"/>
    <rect x="46" y="76" width="8" height="20" rx="4" fill="#4A2C17"/><rect x="86" y="76" width="8" height="20" rx="4" fill="#4A2C17"/>
    <ellipse cx="70" cy="90" rx="24" ry="24" fill="{SKIN}" stroke="{INK}" stroke-width="3"/>
    <circle cx="59" cy="88" r="9.5" fill="#fff" fill-opacity=".75" stroke="{INK}" stroke-width="3"/>
    <circle cx="81" cy="88" r="9.5" fill="#fff" fill-opacity=".75" stroke="{INK}" stroke-width="3"/>
    <path d="M68.5 88 H71.5" stroke="{INK}" stroke-width="3"/>
    <circle cx="59" cy="88" r="2.6" fill="{INK}"/><circle cx="81" cy="88" r="2.6" fill="{INK}"/>
    <circle cx="70" cy="98" r="3.2" fill="#E8B98A"/>
  </g>"""


def waldo_figure_svg() -> str:
    """Full-body, waving Waldo (original simplified drawing): striped hat and shirt, round glasses, jeans."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 150 268">
  <defs>{_STRIPES}</defs>
  <rect x="38" y="190" width="30" height="54" rx="9" fill="{JEANS}" stroke="{INK}" stroke-width="3"/>
  <rect x="76" y="190" width="30" height="54" rx="9" fill="{JEANS}" stroke="{INK}" stroke-width="3"/>
  <ellipse cx="50" cy="248" rx="21" ry="10" fill="#5A3A22" stroke="{INK}" stroke-width="3"/>
  <ellipse cx="96" cy="248" rx="21" ry="10" fill="#5A3A22" stroke="{INK}" stroke-width="3"/>
  <path d="M100 122 L120 86 L134 94 L112 138 Z" fill="url(#stripes)" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>
  <circle cx="127" cy="80" r="9" fill="{SKIN}" stroke="{INK}" stroke-width="3"/>
  <path d="M40 124 L22 172 L36 178 L52 136 Z" fill="url(#stripes)" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>
  <circle cx="28" cy="180" r="9" fill="{SKIN}" stroke="{INK}" stroke-width="3"/>
  <rect x="61" y="108" width="18" height="14" fill="{SKIN}" stroke="{INK}" stroke-width="3"/>
  <path d="M38 124 Q70 110 102 124 L108 196 L32 196 Z" fill="url(#stripes)" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>
  {_head()}
  <path d="M60 106 Q70 113 80 106" fill="none" stroke="{INK}" stroke-width="3" stroke-linecap="round"/>
</svg>"""


def waldo_peek_svg() -> str:
    """Just the top of Waldo's head and two hands, for peeking over an edge."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="20 10 100 88">
  <defs>{_STRIPES}</defs>
  {_head()}
  <circle cx="44" cy="96" r="8" fill="{SKIN}" stroke="{INK}" stroke-width="3"/>
  <circle cx="96" cy="96" r="8" fill="{SKIN}" stroke="{INK}" stroke-width="3"/>
</svg>"""


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Lilita+One&family=Nunito:wght@400;600;800&display=swap');

:root {{ --red: {RED}; --ink: {INK}; --paper: {PAPER}; }}

.stApp {{ background: var(--paper); color: var(--ink); font-family: 'Nunito', 'Trebuchet MS', sans-serif; }}
[data-testid="stHeader"] {{ background: var(--paper); border-bottom: 3px solid var(--ink); }}
.block-container {{ padding-top: 1rem !important; max-width: 1200px; }}
hr {{ border: 0; border-top: 2px solid var(--ink) !important; }}

/* top awning: vertical red/white stripes, like the shirt seen from far away */
.wf-topbar {{
  height: 22px; margin: 0 -4rem 1.1rem;
  background: repeating-linear-gradient(90deg, var(--red) 0 26px, #fff 26px 52px);
  border-bottom: 3px solid var(--ink); border-top: 3px solid var(--ink);
}}

/* hero card */
.wf-hero {{
  position: relative; display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem;
  background: #fff; border: 3px solid var(--ink); border-radius: 20px; box-shadow: 8px 8px 0 var(--red);
  padding: 1.6rem 2rem 0; margin: 0.6rem 0 1.4rem;
}}
.wf-hero-text {{ padding-bottom: 1.6rem; max-width: 720px; }}
.wf-kicker {{ font-family: 'Lilita One', sans-serif; letter-spacing: .18em; font-size: .85rem; color: var(--red); text-transform: uppercase; }}
.wf-title {{
  font-family: 'Lilita One', 'Trebuchet MS', sans-serif !important; font-weight: 400 !important;
  font-size: clamp(2.4rem, 6vw, 4.2rem) !important; line-height: 1.02 !important; margin: .15rem 0 .5rem !important;
  padding: 0 !important; color: var(--red) !important; text-shadow: 3px 3px 0 var(--ink); letter-spacing: .01em;
}}
.wf-title em {{ font-style: normal; color: #fff; -webkit-text-stroke: 2px var(--ink); text-shadow: 3px 3px 0 var(--ink); }}
.wf-sub {{ font-size: 1.08rem; line-height: 1.5; margin: 0; max-width: 620px; }}
.wf-hero-waldo {{ height: 210px; margin-top: -34px; flex: none; transform: rotate(3deg); transform-origin: bottom center; }}
@media (max-width: 760px) {{ .wf-hero-waldo {{ display: none; }} .wf-topbar {{ margin: 0 -1rem 1rem; }} }}

/* section headings become red banner pills */
[data-testid="stHeading"] h3, [data-testid="stHeading"] h2 {{
  display: inline-block; font-family: 'Lilita One', sans-serif !important; font-weight: 400 !important;
  letter-spacing: .05em; font-size: 1.25rem !important; color: #fff !important; background: var(--red);
  border: 3px solid var(--ink); border-radius: 999px; padding: .15rem 1.2rem !important; box-shadow: 3px 3px 0 var(--ink);
  margin-top: .6rem;
}}

/* pictures look like taped-in photos */
[data-testid="stImage"] img {{
  border: 3px solid var(--ink); border-radius: 8px; box-shadow: 6px 6px 0 var(--red); background: #fff;
}}
[data-testid="stImageCaption"] {{ font-family: 'Lilita One', sans-serif; letter-spacing: .03em; color: var(--ink); opacity: .85; }}

/* buttons: chunky pills */
.stButton > button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {{
  font-family: 'Lilita One', sans-serif; letter-spacing: .05em; font-size: 1.1rem; border-radius: 999px;
  border: 3px solid var(--ink); box-shadow: 4px 4px 0 var(--ink); padding: .45rem 1.8rem; transition: transform .08s, box-shadow .08s;
}}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{ background: var(--red) !important; color: #fff !important; }}
.stButton > button:hover, [data-testid="stBaseButton-primary"]:hover {{ transform: translate(-1px, -1px); box-shadow: 6px 6px 0 var(--ink); }}
.stButton > button:active {{ transform: translate(3px, 3px); box-shadow: 1px 1px 0 var(--ink); }}

/* uploader: a dashed "drop the page here" tray */
[data-testid="stFileUploaderDropzone"] {{ background: #fff; border: 3px dashed var(--red); border-radius: 16px; }}
[data-testid="stFileUploader"] label p {{ font-family: 'Lilita One', sans-serif; font-size: 1.15rem; letter-spacing: .03em; }}

/* labels, alerts, verdict */
[data-testid="stWidgetLabel"] p {{ font-weight: 800; }}
[data-testid="stAlert"] {{ border: 2px solid var(--ink); border-radius: 12px; background: #fff; }}
.wf-verdict {{
  background: #fff; border: 3px solid var(--ink); border-left: 14px solid var(--red); border-radius: 12px;
  padding: .7rem 1rem; margin: .4rem 0 .8rem; font-size: 1.05rem;
}}
.wf-verdict b {{ font-family: 'Lilita One', sans-serif; font-weight: 400; color: var(--red); letter-spacing: .03em; font-size: 1.2rem; }}

/* a stripe divider to use between sections */
.wf-stripes {{ height: 12px; margin: 1.4rem 0 .6rem; border: 2px solid var(--ink); border-radius: 999px;
  background: repeating-linear-gradient(90deg, var(--red) 0 18px, #fff 18px 36px); }}

/* a tiny Waldo hiding in the corner of the page, as one does */
.wf-hidden {{ position: fixed; right: 22px; bottom: -6px; width: 54px; z-index: 5; cursor: help; transition: transform .2s; }}
.wf-hidden:hover {{ transform: translateY(-14px); }}
@media (max-width: 760px) {{ .wf-hidden {{ display: none; }} }}
"""


def apply_theme_html() -> str:
    return f"<style>{CSS}</style>"


def hero_html() -> str:
    return f"""
<div class="wf-topbar"></div>
<div class="wf-hero">
  <div class="wf-hero-text">
    <div class="wf-kicker">★ The world-famous search party ★</div>
    <h1 class="wf-title">Where's <em>Waldo</em>?</h1>
    <p class="wf-sub">Upload a wonderfully crowded page and let the computer squint at it for you. On the pages we
    tested, its first guess was right about half the time &mdash; which is, honestly, better than most of us.</p>
  </div>
  <img class="wf-hero-waldo" alt="Cartoon Waldo waving" src="{_data_uri(waldo_figure_svg())}">
</div>"""


def hidden_waldo_html() -> str:
    return (
        f'<img class="wf-hidden" alt="A tiny Waldo peeking over the edge of the page" '
        f'title="Found me! ...but the page you uploaded is harder." src="{_data_uri(waldo_peek_svg())}">'
    )


def stripes_divider_html() -> str:
    return '<div class="wf-stripes"></div>'


def verdict_html(top_confidence: float | None) -> str:
    """One-line plain-English read of the best candidate. Model confidence is not a probability, so say so."""
    if top_confidence is None:
        return (
            '<div class="wf-verdict"><b>No sign of him.</b> Nothing scored above the minimum confidence &mdash; '
            "lower the slider, or try a bigger scan.</div>"
        )
    pct = f"{top_confidence * 100:.0f}%"
    if top_confidence >= 0.5:
        return (
            f'<div class="wf-verdict"><b>Look here first!</b> Candidate #1 is the model&rsquo;s strongest guess '
            f"({pct} confident). It can still be wrong &mdash; check the close-ups below.</div>"
        )
    return (
        f'<div class="wf-verdict"><b>Hmm, no sure thing.</b> The best guess is only {pct} confident, so treat '
        "the close-ups below as leads, not answers.</div>"
    )
