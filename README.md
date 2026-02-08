# Water Sort Screenshot Solver

This is a **static single-page app** that:

1. Parses a Water-Sort style puzzle screenshot in the browser using OpenCV.js (WASM) in a Web Worker
2. Reads the bottom powerup badges (retries / shuffles / add-bottles) using Tesseract.js (WASM OCR)
3. Runs a BFS solver to separate colors into single-color bottles

