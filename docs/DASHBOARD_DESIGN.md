I need you to act as a senior UI/UX designer. Your job is to help me update my entire streamlit dashboard. Below is the design viz spec to follow



```
# Africa Growth Explorer

## Terracotta Editorial Design System

### Streamlit UI/UX Refactor Specification

---

## 1. DESIGN DIRECTION

Redesign the entire Africa Growth Explorer application using an **Editorial Analytics** visual language.

The chosen direction is:

**Terracotta Editorial**

The application should feel:

* analytical
* editorial
* intelligent
* warm
* restrained
* premium
* African without relying on stereotypical African visual motifs
* data-first
* research-oriented
* deliberately designed

It should **not** feel:

* like a generic Streamlit dashboard
* like a SaaS admin panel
* like a corporate banking dashboard
* like a generic AI application
* like a dark-mode developer tool
* like a template dashboard
* overly decorative
* overly colorful

The interface should look like a carefully designed **economic research publication translated into an interactive data product**.

The central visual idea is:

> **warm editorial surface + dark ink typography + rich terracotta data accents**

The application remains functional and analytical. This is a visual and interaction redesign, not a change to the underlying modelling logic.

---

# 2. CORE DESIGN PRINCIPLES

## Principle 1: Editorial before dashboard

Use hierarchy, whitespace, typography and composition to create importance.

Do not solve hierarchy by adding more cards, borders, colors, badges or icons.

The page should breathe.

---

## Principle 2: Data is the visual hero

Charts and analytical outputs should feel integrated into the page rather than inserted into white Streamlit containers.

Avoid the typical pattern:

```text
TITLE
[ CARD ][ CARD ][ CARD ]
-------------------------
[ HUGE CHART ]
-------------------------
[ TABLE ]
```

Instead create visual rhythm:

```text
TITLE + INTRODUCTION

[ KEY METRICS ]

SECTION TITLE
short explanatory sentence

[ PRIMARY VISUAL      ][ SECONDARY VISUAL ]

SECTION TITLE
[ supporting analysis ]

SECTION TITLE
[ table / details ]
```

---

## Principle 3: Terracotta is an accent, not a background

Do not flood the application with terracotta.

Terracotta should identify:

* selected states
* primary actions
* highlighted metrics
* key chart series
* important annotations
* active navigation
* warnings where appropriate

Approximately:

* 70% near-white / cream surfaces
* 20% dark ink / neutral tones
* 10% terracotta and supporting accents

---

## Principle 4: Typography creates the personality

The bespoke character should come primarily from typography and spacing, not decorative graphics.

Use:

### Display font

**Instrument Serif**

Use for:

* page titles
* hero statements
* major editorial headings
* large numerical callouts where appropriate

Fallback:

```text
Georgia, "Times New Roman", serif
```

### Interface font

**DM Sans**

Use for:

* body copy
* navigation
* labels
* buttons
* form controls
* table content
* chart annotations
* captions

Fallback:

```text
Arial, sans-serif
```

Do NOT use the default Streamlit font.

---

# 3. TYPOGRAPHIC HIERARCHY

## Display title

Instrument Serif

Desktop:

```text
42px
line-height: 1.05
font-weight: 400
letter-spacing: -0.02em
```

Example:

```text
Project Overview
```

---

## Editorial section title

Instrument Serif:

```text
28px
line-height: 1.15
font-weight: 400
```

Example:

```text
How the model works
```

---

## Card title

DM Sans:

```text
14px
font-weight: 600
letter-spacing: 0
```

---

## Body

DM Sans:

```text
15px
line-height: 1.6
font-weight: 400
```

---

## Small metadata

DM Sans:

```text
12px
line-height: 1.4
font-weight: 500
```

Use muted text.

---

## Large metric

Use:

```text
36px
Instrument Serif
font-weight: 400
```

Example:

```text
0.74
```

The number should feel like an editorial statistic, not a dashboard widget.

---

# 4. COLOR SYSTEM

Use these colors consistently throughout the entire application.

## Base

### Canvas

```text
#FCFAF7
```

Very warm near-white.

This replaces:

```text
#F8F9FA
```

---

### Primary surface

```text
#FFFFFF
```

Use for cards and elevated content.

---

### Soft surface

```text
#F7F0EA
```

Use for:

* secondary panels
* subtle callouts
* scenario sections
* highlighted information

---

## Ink

### Primary ink

```text
#241D19
```

Use for:

* headings
* body text
* primary labels
* chart titles

This is deliberately not pure black.

---

### Secondary ink

```text
#5E524B
```

Use for:

* supporting text
* descriptions
* axis labels
* metadata

---

### Muted

```text
#8B7D74
```

Use sparingly for:

* captions
* secondary metadata
* inactive labels

---

## Borders

### Default border

```text
#E7DDD5
```

### Strong border

```text
#D7C9BF
```

Borders should be subtle.

Never use heavy grey borders around every object.

---

# 5. TERRACOTTA PALETTE

## Primary Terracotta

```text
#C65A35
```

Primary visual accent.

Use for:

* selected navigation
* primary button
* primary data series
* highlighted numbers
* key annotations

---

## Deep Terracotta

```text
#A94325
```

Use for:

* hover states
* active chart emphasis
* strong accent text
* focused controls

---

## Light Terracotta

```text
#E8B39C
```

Use for:

* secondary chart series
* backgrounds
* selected-state fills
* subtle highlights

---

## Terracotta Tint

```text
#F4E4DA
```

Use for:

* information panels
* scenario backgrounds
* selected card backgrounds

---

# 6. SUPPORTING ACCENT COLORS

Do not introduce arbitrary colors.

Use a restrained supporting palette.

### Deep Plum

```text
#59404A
```

Use for secondary analytical series.

### Dusty Rose

```text
#B98278
```

Use for tertiary series.

### Sand

```text
#D9B88C
```

Use for:

* historical context
* comparison lines
* geographic visualization categories

### Olive Grey

```text
#72745F
```

Use very selectively for neutral contextual information.

---

# 7. SEMANTIC COLORS

Semantic colors should remain visually compatible with the editorial palette.

### Positive

```text
#637A5A
```

### Negative

```text
#9B4637
```

### Warning

```text
#B57A32
```

### Critical

```text
#8C332B
```

Do not use bright:

```text
#00FF00
#FF0000
#0000FF
```

or standard Bootstrap colors.

---

# 8. SHAPE LANGUAGE

The application should use **soft editorial geometry**.

Do not use excessive pill-shaped UI.

## Cards

Use:

```text
border-radius: 12px
border: 1px solid #E7DDD5
background: #FFFFFF
```

Avoid:

```text
border-radius: 999px
```

except for very small status indicators.

---

## Buttons

Use:

```text
border-radius: 8px
```

Primary:

```text
background: #C65A35
color: #FFFFFF
```

Secondary:

```text
background: transparent
border: 1px solid #C65A35
color: #A94325
```

Buttons should feel like editorial controls, not SaaS CTAs.

---

# 9. SHADOW SYSTEM

Use very little shadow.

Default:

```text
box-shadow: 0 2px 10px rgba(36, 29, 25, 0.04)
```

Elevated:

```text
box-shadow: 0 6px 24px rgba(36, 29, 25, 0.07)
```

Never use heavy floating shadows.

---

# 10. PAGE BACKGROUND

The entire application background:

```text
#FCFAF7
```

Do not create large grey rectangles behind every section.

The page itself is the canvas.

Cards exist only where they create meaningful grouping.

---

# 11. SIDEBAR DESIGN

The sidebar should feel like an editorial navigation rail.

Background:

```text
#F7F0EA
```

Border-right:

```text
1px solid #E7DDD5
```

Width:

Approximately:

```text
260px
```

---

## Brand

Instead of:

```text
🌍 Africa Growth Explorer
```

use:

```text
AFRICA
GROWTH EXPLORER
```

Typography:

* AFRICA: DM Sans, 11px, uppercase, letter-spacing 0.12em
* GROWTH EXPLORER: Instrument Serif, 20px

No emoji.

---

## Navigation

Current navigation labels:

```text
Project Overview
Explore Africa
Model Performance
Scenario Explorer
```

Keep these labels.

Do not add emojis.

Active item:

```text
background: #C65A35
color: #FFFFFF
border-radius: 8px
```

Inactive:

```text
color: #5E524B
background: transparent
```

Hover:

```text
background: #F4E4DA
```

---

# 12. COUNTRY SELECTOR

The country selector is a core interaction.

It should be visually prominent but understated.

Label:

```text
COUNTRY
```

Small uppercase DM Sans.

Selector:

white background

border:

```text
1px solid #D7C9BF
```

Focused state:

```text
border-color: #C65A35
```

Do not use bright blue Streamlit focus states.

---

# 13. PAGE HEADER

Every page should begin with a strong editorial hierarchy.

Pattern:

```text
PROJECT OVERVIEW

Africa Growth Explorer
Predicting near-term GDP per capita growth across African countries using
World Bank development indicators.
```

Do not repeat title information unnecessarily.

Avoid:

```text
📋 Project Overview
### Africa Growth Explorer...
```

The emoji and Markdown heading hierarchy currently make the app feel like an instructional notebook rather than a finished product.

---

# 14. HERO AREA

The Overview page should begin with an editorial hero.

Recommended structure:

Left:

```text
AFRICA GROWTH EXPLORER

Predicting near-term GDP per capita growth
across African countries.

A machine-learning decision-support system built
from World Bank Development Indicators.
```

Right:

A subtle abstract Africa visual or geographic data motif.

The Africa graphic must remain understated.

Do not use a stock illustration.

Do not use a stereotypical tribal pattern.

Do not use an AI-generated "futuristic Africa" illustration.

Preferred:

* geographic outline
* subtle dot field
* contour lines
* data points
* restrained terracotta map

---

# 15. KPI SYSTEM

The current application uses metrics for:

* MAE
* RMSE
* R²
* directional accuracy
* country indicators
* scenario predictions

These should all use the same visual language.

Each KPI:

```text
LABEL
36px value
short explanatory line
```

Example:

```text
TEST R²

0.742

Model fit on held-out test data
```

Do not render every KPI as a visually identical heavy card.

Use whitespace and thin borders.

---

# 16. KPI COLOR RULE

Not every number should be terracotta.

Default:

```text
Ink
```

Use terracotta only for the most important metric on the page.

For example:

```text
R²
0.742
```

can use terracotta.

The other values remain dark ink.

This keeps the accent meaningful.

---

# 17. "HOW IT WORKS" COMPONENT

Replace the current bullet-heavy presentation with an editorial process strip.

Five stages:

```text
01  COLLECT
World Bank indicators

02  ENGINEER
Lagged features and transformations

03  TRAIN
Machine-learning model

04  PREDICT
Near-term growth

05  EXPLORE
Scenarios and comparisons
```

Use thin horizontal connecting rules.

Number circles:

terracotta outline or light terracotta fill.

Do not use generic AI icons.

If icons are used, use simple line icons from a consistent icon system.

---

# 18. CHART DESIGN LANGUAGE

This is one of the most important parts of the redesign.

The charts must not look like default Matplotlib charts.

Every chart should share the same styling.

Background:

```text
transparent
```

Figure face:

```text
#FFFFFF
```

or transparent where appropriate.

Spines:

Remove top and right spines.

Use very subtle bottom and left spines.

Grid:

Horizontal grid only where useful.

Color:

Primary:

```text
#C65A35
```

Secondary:

```text
#59404A
```

Tertiary:

```text
#B98278
```

Historical / contextual:

```text
#D9B88C
```

---

# 19. CHART TYPOGRAPHY

Chart title:

DM Sans semibold

```text
14px
```

Axis labels:

DM Sans

```text
11px
```

Tick labels:

DM Sans

```text
10px
```

Do not use large bold chart titles.

The page section heading carries the hierarchy.

---

# 20. CHART GRID

Avoid dense grids.

Use:

```text
alpha ≈ 0.12
```

Grid lines should almost disappear.

The data should dominate.

---

# 21. LINE CHARTS

Use:

```text
linewidth = 2.2
```

No excessive markers.

Markers should only appear when individual observations matter.

Observed historical:

```text
#C65A35
```

Prediction / target:

```text
#59404A
```

Use dashed lines sparingly.

Do not make every series dotted or dashed.

---

# 22. ACTUAL VS PREDICTED

Use a clean scatter plot.

Actual:

x-axis

Predicted:

y-axis

Use:

```text
point color = #C65A35
alpha ≈ 0.65
```

Reference diagonal:

```text
#8B7D74
linewidth = 1
linestyle = "--"
```

Do not use a large legend if the chart title and axis labels already explain the visual.

---

# 23. RESIDUAL PLOT

Primary points:

```text
#59404A
```

Zero line:

```text
#C65A35
```

Keep annotation minimal.

The chart should answer:

> Is the error pattern random or systematic?

Do not decorate it unnecessarily.

---

# 24. FEATURE IMPORTANCE

Use horizontal bars.

Primary:

```text
#C65A35
```

Non-significant features:

```text
#D9D0CA
```

Significant features should visually dominate.

Sort descending.

Use value labels only when they materially improve readability.

Do not create a rainbow palette for feature importance.

---

# 25. COUNTRY / AFRICA MAP

The map should use a terracotta tonal scale.

Light:

```text
#F4E4DA
```

Medium:

```text
#E8B39C
```

Strong:

```text
#C65A35
```

Deep:

```text
#A94325
```

No rainbow maps.

No blue choropleth.

No default geopandas / matplotlib color map.

Legend should use editorial typography.

---

# 26. TABLE DESIGN

Tables should resemble research tables rather than spreadsheet widgets.

Header:

```text
background: #F7F0EA
color: #5E524B
font-weight: 600
```

Rows:

white / transparent.

Borders:

very subtle.

Use horizontal dividers rather than full cell boxes.

Numeric columns:

right aligned.

Text columns:

left aligned.

Avoid zebra striping unless the table is very dense.

---

# 27. DATAFRAME RULE

Never expose raw feature codes unless the user explicitly needs technical metadata.

For example:

Avoid displaying:

```text
EG.ELC.ACCS.ZS
```

as the main visible label.

Display:

```text
Electricity access (%)
```

Technical codes may appear in a secondary metadata column or expandable detail section.

---

# 28. ALERTS AND GUARDRAILS

Warnings are important to this application and should not look like standard Streamlit yellow boxes.

Use editorial callout panels.

### Information

Background:

```text
#F7F0EA
```

Accent:

```text
#C65A35
```

### Warning

Background:

```text
#FAF1E2
```

Accent:

```text
#B57A32
```

### Critical

Background:

```text
#F5E3DF
```

Accent:

```text
#8C332B
```

The causal disclaimer should be visually distinctive but calm.

---

# 29. CAUSAL GUARDRAIL

This is one of the most important elements in the product.

Do not hide it inside a generic Streamlit warning.

Present it as an editorial research note.

Example structure:

```text
CAUSAL GUARDRAIL

These predictions describe statistical associations in the
model. They do not estimate the causal effect of changing
an individual development indicator.

Use scenario results as analytical evidence, not as
cause-and-effect claims.
```

Use a thin terracotta vertical rule.

No warning emoji.

No siren icon.

No AI icon.

---

# 30. SCENARIO EXPLORER

The Scenario Explorer should feel like a research instrument.

The hierarchy should be:

```text
SCENARIO EXPLORER

Country + reference year

CURRENT PROFILE

[ indicator controls ]

SCENARIO RESULT

Baseline       Scenario       Difference

INDICATORS DRIVING THE CHANGE

[ ranked table ]

CAUSAL GUARDRAIL
```

Do not make the slider controls visually overpowering.

---

# 31. SLIDER DESIGN

Use the native Streamlit sliders only if CSS can make them conform.

Track:

```text
background: #E7DDD5
```

Active track:

```text
#C65A35
```

Thumb:

```text
#C65A35
```

Value:

dark ink.

Each slider should have:

```text
INDICATOR NAME
current value
slider
small range / context note
```

Avoid excessive explanatory text directly beneath every control.

Use help tooltips for technical details.

---

# 32. SCENARIO RESULTS

Three columns:

```text
BASELINE

2.41%

SCENARIO

3.08%

CHANGE

+0.67 pp
```

The scenario value can use terracotta.

The change uses semantic color.

Do not automatically assume positive is green.

Use muted semantic colors within the palette.

---

# 33. INDICATORS DRIVING CHANGE

The current table should become an editorial analytical table.

Columns:

```text
Indicator
Baseline
Scenario
Change
Model response
```

Emphasize the "Model response" column.

Use tiny horizontal directional indicators if helpful.

Do not use decorative icons.

---

# 34. EXPANDERS

Streamlit expanders should not look like grey dropdown boxes.

Style them as subtle editorial disclosure rows.

Example:

```text
+ View technical details
```

Use understated border and whitespace.

Expanded state:

white background.

---

# 35. MODEL PERFORMANCE PAGE

This page should feel like a research appendix.

Hierarchy:

```text
MODEL PERFORMANCE

What the model gets right
What it gets wrong
How it compares with simple baselines
```

Then:

```text
MODEL VERDICT
```

Then:

```text
BASELINE COMPARISON

ACTUAL VS PREDICTED
RESIDUAL ANALYSIS
FEATURE IMPORTANCE
PERFORMANCE BY YEAR
LIMITATIONS
```

Do not lead with a wall of metrics.

Lead with the interpretation.

---

# 36. MODEL VERDICT

The current statistical significance finding should be visually prominent.

Use:

```text
MODEL VERDICT

The model is not statistically distinguishable from
the global-mean baseline on the held-out test set.
```

Then provide the supporting metrics below.

This should feel like a research finding, not an error message.

---

# 37. LIMITATIONS

Present limitations as numbered editorial notes rather than a bullet dump.

Example:

```text
01
PARITY WITH THE BASELINE

The model does not demonstrate statistically significant
improvement over the global mean.

02
TEMPORAL GENERALIZATION

...

03
ASSOCIATION ≠ CAUSATION

...
```

This gives the research limitations proper hierarchy.

---

# 38. OVERVIEW PAGE STRUCTURE

Recommended final composition:

```text
AFRICA GROWTH EXPLORER

Large editorial headline
Short description

[ 4 primary metrics ]

HOW IT WORKS
01 → 02 → 03 → 04 → 05

KEY INDICATORS DRIVING THE PREDICTION
[ bar chart ]

DATA COVERAGE
[ Africa map ]

MODEL NOTE
[ concise research note ]

INTENDED USERS

CAUSAL GUARDRAIL
```

---

# 39. EXPLORE AFRICA PAGE STRUCTURE

Use:

```text
EXPLORE AFRICA

[ COUNTRY SELECTOR ]

Kenya
Development indicators over time

[ primary GDP growth chart ]

CURRENT PROFILE

[ KPI ][ KPI ][ KPI ]
[ KPI ][ KPI ][ KPI ]

INDICATOR TRENDS

[ selector ]

[ chart ]

REGIONAL COMPARISON

[ research table ]
```

The selected country should feel like the subject of a research profile.

---

# 40. MODEL PERFORMANCE PAGE STRUCTURE

Use:

```text
MODEL PERFORMANCE

Research note / verdict

[ MAE ][ RMSE ][ R² ][ Directional accuracy ]

BASELINE COMPARISON

[ table ]

ACTUAL VS PREDICTED

[ chart ]

RESIDUAL ANALYSIS

[ chart ]

FEATURE IMPORTANCE

[ chart ]

PERFORMANCE BY YEAR

[ table / chart ]

LIMITATIONS

[ editorial notes ]
```

---

# 41. SCENARIO PAGE STRUCTURE

Use:

```text
SCENARIO EXPLORER

Country
Reference year

CURRENT PROFILE
[ observed feature summary ]

ADJUST THE SCENARIO

[ controls ]

PREDICTION

[ baseline ][ scenario ][ difference ]

WHAT MOVED THE PREDICTION?

[ analytical table ]

CAUSAL GUARDRAIL
```

---

# 42. NAVIGATION LANGUAGE

Use plain analytical language.

Keep:

```text
Project Overview
Explore Africa
Model Performance
Scenario Explorer
```

Do not add:

```text
AI Insights
Smart Forecasts
AI Copilot
Intelligent Analytics
Ask the Model
Magic Analysis
```

The product should communicate what it actually does.

---

# 43. ICONOGRAPHY

**No AI icons.**

Do not use:

* robot heads
* sparkle icons
* magic wands
* neural network icons
* chatbot icons
* brains
* futuristic circuit icons
* glowing stars
* "AI" badges
* emoji

Do not use emojis anywhere in page titles, navigation, metric labels, button labels, alerts, section headings, or explanatory copy.

The current application contains emoji-based headings such as Project Overview, Explore Africa, Model Performance and Scenario Explorer. Replace these with typography and subtle line icons where necessary.

Icons should be:

* minimal
* monochrome
* thin-line
* functional
* consistent

Preferred icon style:

Lucide-style line icons.

Use icons only when they improve scanning.

Never use an icon simply to decorate a heading.

---

# 44. COPY STYLE

Copy should sound like a human researcher wrote it.

Use:

* direct language
* plain English
* short sentences
* specific wording
* analytical terminology where appropriate
* restrained confidence

Avoid:

* "unlock"
* "empower"
* "supercharge"
* "revolutionize"
* "seamlessly"
* "cutting-edge"
* "next-generation"
* "leverage"
* "harness"
* "game-changing"
* "intelligent"
* "AI-powered"
* "transformative"

---

# 45. NO AI-SOUNDING COPY

Do not rewrite existing analytical content into generic marketing language.

For example:

Bad:

```text
Unlock powerful AI-driven insights into Africa's economic future.
```

Better:

```text
Estimate near-term GDP per capita growth from observed
development indicators.
```

The application is a serious analytical tool.

Write like an economist / data scientist, not a SaaS marketer.

---

# 46. NO EM DASHES

The entire application must avoid em dashes.

Do not use:

```text
—
```

Use:

```text
-
```

or rewrite the sentence.

This applies to:

* page copy
* chart titles
* captions
* warnings
* tooltips
* table labels
* generated strings
* helper text
* code comments where user-facing text is involved

---

# 47. NO GENERIC AI COPY PATTERNS

Avoid phrasing such as:

```text
"At a glance..."
"Let's dive in..."
"Here's what you need to know..."
"Powered by..."
"Your journey..."
"Discover..."
"Unlock..."
```

Use analytical headings instead:

```text
Model performance
Observed growth
Regional comparison
Feature importance
Scenario response
Model limitations
```

---

# 48. SPACING SYSTEM

Use an 8px base spacing system.

```text
4px   micro
8px   xs
12px  sm
16px  md
24px  lg
32px  xl
48px  2xl
64px  3xl
```

Most sections should use:

```text
24px to 40px
```

between blocks.

Avoid tightly packed components.

---

# 49. CONTENT WIDTH

Keep the main analytical content visually constrained.

Target:

```text
max-width: 1400px
```

For editorial text:

```text
max-width: 780px
```

Do not allow long paragraphs to span the entire dashboard.

---

# 50. RESPONSIVE BEHAVIOUR

Desktop:

* sidebar visible
* two-column analytical layouts
* charts side by side where appropriate

Tablet:

* reduce card widths
* collapse multi-column arrangements

Mobile:

* stacked cards
* full-width charts
* sidebar collapses
* no horizontal overflow
* tables scroll horizontally if necessary

Do not simply shrink everything.

Reflow the hierarchy.

---

# 51. STREAMLIT CSS STRATEGY

Use a single centralized CSS injection function.

Example conceptual structure:

```python
def inject_editorial_styles():
    st.markdown(
        """
        <style>
            ...
        </style>
        """,
        unsafe_allow_html=True
    )
```

Do not scatter CSS throughout the application.

Create design tokens at the top:

```python
COLORS = {
    "canvas": "#FCFAF7",
    "surface": "#FFFFFF",
    "surface_soft": "#F7F0EA",
    "ink": "#241D19",
    "ink_secondary": "#5E524B",
    "muted": "#8B7D74",
    "border": "#E7DDD5",
    "border_strong": "#D7C9BF",
    "terracotta": "#C65A35",
    "terracotta_deep": "#A94325",
    "terracotta_light": "#E8B39C",
    "terracotta_tint": "#F4E4DA",
    "plum": "#59404A",
    "rose": "#B98278",
    "sand": "#D9B88C",
    "positive": "#637A5A",
    "warning": "#B57A32",
    "critical": "#8C332B",
}
```

Use these tokens everywhere.

Never hardcode a new random color later.

---

# 52. MATPLOTLIB STYLE

Create a single project plotting style.

Conceptually:

```python
def set_editorial_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "#D7C9BF",
        "axes.labelcolor": "#5E524B",
        "xtick.color": "#5E524B",
        "ytick.color": "#5E524B",
        "text.color": "#241D19",
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "grid.color": "#E7DDD5",
        "grid.alpha": 0.45,
    })
```

Remove top and right spines where possible.

Do not use Seaborn default styling.

Every project chart must use the same style.

---

# 53. CHART COLOR SEQUENCE

Use this fixed sequence:

```python
CHART_COLORS = [
    "#C65A35",
    "#59404A",
    "#B98278",
    "#D9B88C",
    "#72745F",
]
```

However, do not automatically use five colors.

Prefer one or two colors per chart.

Use multiple colors only when the data has a real categorical distinction.

---

# 54. CHART CONTAINER RULE

Do not wrap every chart in a giant card with a heavy border.

Preferred:

```text
SECTION TITLE
short explanation

[ chart ]
```

For a secondary visualization:

```text
[ chart card ]
```

The chart itself should visually integrate with the page.

---

# 55. INTERACTION STATES

Every interactive component needs consistent states.

Default:

neutral border

Hover:

light terracotta tint

Focus:

terracotta border

Active:

deep terracotta

Disabled:

muted text / washed-out border

Never use default Streamlit bright blue focus styling.

---

# 56. TOOLTIPS

Technical information belongs in tooltips rather than visible copy when possible.

Example:

Visible:

```text
R²
0.742
```

Tooltip:

```text
Coefficient of determination on the held-out test set.
```

This maintains visual clarity without hiding useful technical detail.

---

# 57. TECHNICAL DETAIL HIERARCHY

Do not remove technical information.

Instead classify it:

### Primary

Information required to understand the result.

### Secondary

Useful technical context.

### Tertiary

Implementation detail.

Primary information stays visible.

Secondary information goes into captions or supporting text.

Tertiary information goes into expanders.

---

# 58. PRESERVE THE EXISTING ANALYTICAL INTEGRITY

Do not alter the underlying statistical meaning while redesigning the UI.

Do not change:

* model outputs
* feature calculations
* training/test logic
* scenario calculations
* extrapolation guardrails
* significance interpretation
* baseline comparisons
* feature responsiveness logic
* causal disclaimers

The application currently explicitly distinguishes predictive association from causal inference. Preserve that distinction exactly in substance.

---

# 59. IMPORTANT SCENARIO EXPLORER RULE

The current scenario implementation intentionally excludes features that do not actually move the prediction for the selected country-year.

Do not remove that behaviour merely because it makes the UI more complicated.

Instead explain it elegantly.

Suggested UI:

```text
WHY THESE INDICATORS?

The controls below are limited to indicators that
change the model's prediction for this country-year.
```

Then:

```text
View excluded indicators
```

inside an expander.

This preserves the underlying analytical guardrail while improving comprehension.

---

# 60. OVERALL VISUAL TEST

After implementation, the application should pass this test:

When someone sees a screenshot without the browser chrome, it should be immediately recognizable as:

**Africa Growth Explorer**

and should look like a bespoke economic research product.

It should NOT look like:

* a Streamlit template
* Power BI
* Tableau
* a generic fintech dashboard
* an AI chatbot
* a generic data science notebook

---

# 61. THE VISUAL PERSONALITY IN ONE SENTENCE

> A warm editorial research interface that uses terracotta as a precise analytical accent against a near-white canvas, with distinctive serif typography, restrained data visualization, and strong research hierarchy.

---

# 62. IMPLEMENTATION PRIORITY

Refactor in this order:

### 1. Global foundation

* fonts
* background
* typography
* colors
* sidebar
* navigation
* buttons
* inputs
* cards
* tables

### 2. Chart system

* Matplotlib theme
* chart typography
* chart colors
* gridlines
* annotations
* Africa map styling

### 3. Overview page

Establish the full visual language here first.

### 4. Explore Africa

Apply the same language without creating a separate visual system.

### 5. Model Performance

Make it feel more like a research appendix.

### 6. Scenario Explorer

Make interaction feel like a research instrument.

### 7. Final polish

* remove emojis
* remove AI iconography
* remove generic copy
* remove inconsistent colors
* remove default Streamlit styling
* remove unnecessary borders
* check responsive behaviour
* check spacing
* check all chart titles and labels
* check all warnings and disclaimers

---

# 63. NON-NEGOTIABLES

The implementation AI must follow these rules:

1. **No emojis in the UI.**
2. **No AI-themed icons.**
3. **No em dashes.**
4. **No generic AI marketing language.**
5. **No bright blue Streamlit defaults.**
6. **No rainbow chart palettes.**
7. **No arbitrary colors outside the design tokens.**
8. **No unnecessary card nesting.**
9. **No excessive rounded pills.**
10. **No changing analytical logic while refactoring the UI.**
11. **No removing analytical warnings or caveats.**
12. **No rewriting the product into a generic SaaS dashboard.**
13. **Use Instrument Serif + DM Sans consistently.**
14. **Terracotta is an accent, not the page background.**
15. **The near-white canvas must remain dominant.**
16. **Charts must share one coherent visualization language.**
17. **Technical detail should be progressively disclosed rather than deleted.**
18. **Every page must feel like part of the same product.**

---

# 64. FINAL CREATIVE REFERENCE

Think:

**Financial Times editorial design**
+
**modern economic research publication**
+
**interactive analytical instrument**

Not:

**generic AI dashboard**
+
**startup SaaS**
+
**Power BI clone**

The interface should feel like someone deliberately designed a data product for serious analytical work.

The result should be **quietly distinctive** rather than visually loud.
```