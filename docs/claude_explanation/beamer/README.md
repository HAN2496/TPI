# Bounce Rate Reconstruction Beamer

Build from this directory with:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The 12-slide deck is based on `../experiment.md` and summarizes the current
seed-42, driver-held-out results for Matern 3/2 KF, KF + LSTM, and the six
online/offline model-free networks. Figures are loaded from `assets/`.
