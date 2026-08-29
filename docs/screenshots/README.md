# App screenshots — public deploy

The app is publicly deployed at [https://africa-growth-ml.streamlit.app/](https://africa-growth-ml.streamlit.app/). Four dashboard screenshots from the live app are committed under `assets/` and referenced in the README and capstone report.

Screenshots included:

- `assets/project_overview.png` — Project Overview page
- `assets/explore_africa.png` — Explore Africa indicator trends
- `assets/model_performance.png` — Model Performance by year
- `assets/scenario_explorer.png` — Scenario Explorer adjustments

Locally verified:

```bash
pip install -r requirements.txt
streamlit run app.py --server.headless true   # then open http://localhost:8501
```
