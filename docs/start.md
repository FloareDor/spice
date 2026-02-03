Here is the comprehensive **Master Plan** for your project. This document is designed to serve as your "North Star" for development, ensuring you stay focused on the high-value features that will make this tool essential for producers.

---

# Project Blueprint: "LocalVibe" (Working Title)

**The "Splice" Experience for Local Libraries. Private. Offline. Semantic.**

## 1. Executive Summary

**The Problem:** Producers have terabytes of samples (folders named `New Folder (2)`, `Cymatics_Vol_9`) but can't find anything. Existing tools are either "dumb" (filename search only), "bloated" (Waves Cosmos), or "cloud-tethered" (Splice Bridge).
**The Solution:** An ultra-lightweight, open-source desktop app that indexes a user's existing local library. It uses **state-of-the-art AI** to understand the "vibe" of a sound (via text search) and **DSP** to understand the "theory" (BPM, Key).
**The Core Philosophy:** "Your sounds, your drive, your privacy. No cloud required."

---

## 2. User Stories (The Experience)

### Core Loop: The "Flow State" Search

> **User:** Ravi (Producer)
> **Goal:** Find a specific texture for a beat.

1. **Trigger:** Ravi opens the app (Instant load).
2. **Action:** He types *"Dusty lo-fi piano chords, sad vibe"* into the search bar.
3. **Result:** The app returns 50 samples sorted by relevance, pulling from three different sample packs on his hard drive.
4. **Refinement:** He clicks a filter pill: `Key: F Minor`. The list shrinks to 8 perfect matches.
5. **Audition:** He presses `Spacebar` to preview. The waveform visualizes the transient peaks.
6. **The "Kill Shot":** He clicks and drags the file directly from the app into **Ableton Live**. It loads instantly on the timeline.

### Secondary Story: The "Smart Import"

> **User:** Ravi adds a new "Unsorted" folder of 5GB of WAVs.

1. **Action:** Drags folder into the app.
2. **Process:** The app scans immediately. Files appear instantly with filenames.
3. **Background:** Over the next few minutes, a small status bar shows "Analyzing Vibe...".
4. **Outcome:** 10 minutes later, that "Unsorted" folder is fully searchable by BPM, Key, and Instrument tags.

---

## 3. The Tech Stack (Optimized for "Little GPUs")

This stack is chosen for maximum performance on standard consumer hardware (MacBook Air, mid-range Windows laptops) without requiring a dedicated NVIDIA GPU.

### **Frontend & Application Shell**

* **Framework:** **Python + PyQt6** (or PySide6).
* *Why:* The **only** reliable way to implement native OS-level "Drag-and-Drop" into DAWs (via `QMimeData`). Electron/Web apps struggle with this security sandbox.


* **UI Library:** **Custom Widgets** (Canvas for waveform).
* *Why:* Keep it lightweight. No heavy HTML/CSS rendering engine.



### **The "Brain" (AI & Signal Processing)**

* **Semantic Search (Text-to-Audio):** **CLaMP 3 (Contrastive Language-Music Pre-training)**.
* *Optimization:* Convert model to **ONNX Format (Int8 Quantized)**.
* *Why:* beats LAION-CLAP for musical nuance (chords, textures, genre).


* **Music Theory (BPM, Key, Rhythm):** **Essentia (C++, Python)**.
* *Why:* Industry standard accuracy. Runs purely on CPU. Blazing fast (<10ms per file).


* **Auto-Tagging (Zero-Shot):** **Embedding Comparison**.
* *Method:* Compare the audio embedding against a pre-computed list of 500 "Splice Tags" (e.g., "Trap", "Warm", "Punchy"). Top 5 matches become tags.
* *Why:* Gives you "AI Tagging" without running a heavy captioning LLM.



### **Data & Storage**

* **Vector Database:** **LanceDB**.
* *Why:* Serverless, runs in-process, stores data in a single file on disk. Extremely fast for vector lookups.


* **Metadata Store:** **SQLite**.
* *Why:* Reliable storage for file paths, BPM, Key, Duration, and user favorites.



---

## 4. Optimization Strategy ("The Secret Sauce")

How to make Python feel like C++:

1. **The "Waterfall" Indexing Pipeline:**
* **Phase 1 (Instant):** Scan filenames & file sizes. Show them in UI immediately.
* **Phase 2 (Fast):** Run Essentia (CPU) for Duration/Waveform.
* **Phase 3 (Heavy):** Run ONNX Model (NPU/GPU) for Embeddings. *Queue this to run when the computer is idle.*


2. **Audio Fingerprinting:**
* Don't embed the whole file. Analyze **10-second chunks** (Start + Middle).
* *Speedup:* 6x faster indexing.


3. **Lazy Model Loading:**
* Do **not** load the 200MB AI model on startup.
* Load it only when the user types a query or adds a new folder. Keep startup time < 2 seconds.



---

## 5. Competitive Advantage (The Moat)

| Feature | Your Tool ("LocalVibe") | Waves Cosmos | Sononym | Splice Desktop |
| --- | --- | --- | --- | --- |
| **Search Type** | **"Vibe" (Semantic Text)** | Tags Only | Audio Similarity | Tags Only |
| **Privacy** | **100% Local / Offline** | Login Required | Local | Cloud Tethered |
| **BPM/Key** | **AI Detected** | AI Detected | AI Detected | Database Lookup |
| **Integration** | **Direct Drag-to-DAW** | Direct Drag-to-DAW | Direct Drag-to-DAW | Bridge Plugin (Annoying) |
| **Price** | **Open Source (Free)** | Paid / Bloatware | Expensive ($99) | Subscription |
| **Flexibility** | **Customizable Models** | Closed Box | Closed Box | Closed Box |

---

## 6. Implementation Roadmap

### **Phase 1: The Skeleton (Day 1-2)**

* Set up Python environment with **PyQt6**.
* Build a basic window that lists `.wav` files from a hardcoded folder.
* Implement `QMimeData` to prove you can drag a file from your window to the Desktop.

### **Phase 2: The "Smart" Layer (Day 3-5)**

* Integrate **Essentia**. Display BPM and Key next to filenames.
* Get **CLaMP 3 (ONNX)** running in a standalone script.
* Verify you can search "Drum" and get a vector result.

### **Phase 3: The Integration (Day 6-7)**

* Connect the UI Search Bar -> ONNX Runtime -> LanceDB.
* Implement the "Waterfall" background thread so the UI doesn't freeze during indexing.

### **Phase 4: Polish (Week 2)**

* Draw the waveform (using `matplotlib` or raw `QPainter` for speed).
* Add the "Zero-Shot" auto-tagging system.
* Package as an `.exe` / `.app` using **PyInstaller**.

---

## 7. Next Immediate Step

**Validate the Core Tech:** Don't build the UI yet. Write a single Python script that:

1. Takes a folder path.
2. Uses **Essentia** to print the BPM of every WAV.
3. Uses **CLaMP 3** to print the vector embedding of every WAV.

*If you can do this script, you have a product. The rest is just buttons.*