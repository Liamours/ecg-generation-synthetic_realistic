# ecg-generation-synthetic_realistic

Synthetic ECG page image generation from PTB-XL signals: pages, panels, and leads with exact labels, recorded transforms, and trace masks. Code, configs, and scripts only; generated datasets go to `datasets/training/`, per the file placement rule in `wiki/overview/directory.md` in the parent project root. Design: `wiki/overview/synthetic_data_flow.md`; current work and decisions: `wiki/TODO.md`.

Main entry point: `scripts/generate_dataset.py`. The panel template redesign in progress is `scripts/prototype_ideal_panel_template.py`.
