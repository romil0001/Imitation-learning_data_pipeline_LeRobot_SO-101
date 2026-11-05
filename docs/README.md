# GitHub Pages showcase

The public output showcase lives at [`index.html`](index.html). Static assets are in [`showcase_assets/`](showcase_assets/) (copies of committed `assets/frames/` and `assets/graphs/`).

## Enable GitHub Pages

1. Open the repo on GitHub → **Settings** → **Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Branch: **main** (or your default), folder: **`/docs`**.
4. Save. After the workflow finishes, the site is at  
   **https://romil0001.github.io/SO101_Visuomotor_Manipulation/**

No build step is required (`index.html` is self-contained).

## Preview locally

```bash
open docs/index.html
# or: python -m http.server 8765 --directory docs
# then visit http://127.0.0.1:8765/
```
