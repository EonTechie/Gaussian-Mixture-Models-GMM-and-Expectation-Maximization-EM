# Gaussian Mixture Models (GMM) and Expectation-Maximization (EM)

## Overview
This repository contains Python implementations of the Expectation-Maximization (EM) algorithm for two tasks as a personal work for a  BLG527E - Machine Learning at ITU:
1. **Gaussian Mixture Models (GMM):** Clustering a 2D dataset with 3 Gaussian clusters.
2. **Coin Flipping Experiments:** Estimating parameters (coin biases and selection probabilities) from simulated coin-flip data.

## Features
- Implementation of the EM algorithm without external toolboxes.
- Visualization of Gaussian contours after clustering.
- Dynamic simulations for coin-flipping with parameter estimation.

## Project Structure
- `gmm_em.py`: Generates a 2D dataset, applies EM to fit a GMM, and visualizes results.
- `em_coin_flip.py`: Simulates coin flips and estimates coin biases using EM.
- `em_updated_coin_flip.py`: Extends the coin-flipping experiment with additional parameters.

## How to Run
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
   [Download the repository directly here.](<repository-link>)

2. Run the desired script:
   ```bash
   python gmm_em.py
   python em_coin_flip.py
   python em_updated_coin_flip.py
   ```

## Dependencies
- Python 3.x
- `numpy`, `matplotlib` (install via `pip install numpy matplotlib`)

## Results
- **GMM:** Displays Gaussian contours after clustering.
- **Coin Flip Experiments:** Prints coin biases and probabilities for each simulation.

## Privacy Note

This repository is anonymized to protect academic and personal details, adhering to privacy and plagiarism policies.