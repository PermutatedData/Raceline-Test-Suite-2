# Raceline Test Suite 2

Building off of **Raceline Test Suite**, a lightweight web-based tool for creating and testing racetracks

Mainly features various basic Delaunay triangulations plus a cone sorting algorithm

---

## ✨ Features
- **Draw Racetracks**  
  Intuitive HTML canvas interface for quickly sketching custom racetracks.
- **Download as CSV**  
  Export track geometry (cones, boundaries, or path) into `.csv` files for use in simulations.
- **Run Custom Raceline**  
  Integrate your raceline generation algorithm and test it directly on the tracks you create.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** installed on your system  
- A modern web browser (Chrome, Firefox, Edge, Safari)  

### Installation
```bash
git clone https://github.com/RichardZhang06/raceline-test-suite.git
cd raceline-test-suite
pip install -r requirements.txt
```

### Usage Workflow
1. Open the track drawer in your brower
```bash
open racetrack_drawer/racetrack_drawer.html       # Mac
xdg-open racetrack_drawer/racetrack_drawer.html   # Linux
start racetrack_drawer/racetrack_drawer.html      # Windows
```
2. Draw your track using the interactive canvas
3. Download the track as a `.csv` file
4. Move the file into the `tracks/` directory
5. Modify `racetest.py` to
- Load your custom track from `tracks/`
- Use your raceline generation algorithm (replace or extend the midline algorithm)
6. Run the test
```bash
python racetest.py
```

### Cone Sorting
The algorithm used is a weighted sum of angles distances with 1-step lookahead. Hard cutoffs are used, likely giving slight performance benefits

Turns out to be very sensitive to weights and other parameters (see Images)
