# App screenshots — why this folder has no PNGs

The remediation plan calls for 2–3 screenshots of the running app in this
folder. **No screenshots are committed because none could be produced
truthfully in this environment**, and a fabricated/placeholder image would
be worse than none:

- This development sandbox has no browser binary and no network route to one
  (Playwright's CDN, Chrome-for-Testing storage, and jsDelivr are all
  unreachable; only GitHub and PyPI respond). A headless screenshot pass is
  therefore impossible here.
- Streamlit Cloud deployment requires interactive account sign-in, which the
  sandbox cannot perform. The app is **not** publicly deployed as part of this
  submission (see `README.md` → Deployment for the honest status).

What **is** verified locally (reproducible on any machine):

```bash
pip install -r requirements.txt
streamlit run app.py --server.headless true   # then open http://localhost:8501
```

Screenshots to capture after deployment (suggested, in this order):

1. `01_home.png` — Home tab: headline metrics (test MAE 1.82, CI spanning
   zero vs global mean) and the parity caveat banner.
2. `02_scenario.png` — Scenario Explorer: Ghana, one slider moved, single
   one-at-a-time delta shown with the significance caveat.
3. `03_limitations.png` — Methodology & Limitations tab with the validation
   vs test table and extrapolation warning (set inflation to ~90 to trigger).

If you run the app locally and want evidence in the repo, save the captures
here as `01_home.png`, `02_scenario.png`, `03_limitations.png` (each < 500 KB,
`optipng -o1` or PNG8 acceptable).
