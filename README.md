# From ODEs to Neural ODEs

## 📘 Overview
This repository explores the transition **from Ordinary Differential Equations (ODEs) to Neural ODEs**, combining theoretical foundations with practical experiments.  
The implementation is adapted from [learner/hnn.py](https://github.com/jpzxshi/learner/blob/master/learner/nn/hnn.py) and extended to demonstrate how neural networks can solve ODE systems of different orders.

---

## 📂 Project Structure
- `slides/`  
  Contains lecture slides explaining the theory and motivation behind Neural ODEs.  
  👉 [Slides PDF](https://github.com/Autumn61q/From_ODEs_to_Neural_ODEs/blob/main/slides/From_ODEs_to_Neural_ODEs.pdf)

- `sympnets/scripts/`  
  Entry-point scripts for experiments:
  - `1order_model_solver.py` — First-order model and solver  
  - `2order_model_solver.py` — Second-order model and solver  
  - `3order_model_solver.py` — Third-order model and solver  

---

## 🚀 Getting Started
Run experiments directly from the scripts:

```bash
# First-order model
python sympnets/scripts/1order_model_solver.py

# Second-order model
python sympnets/scripts/2order_model_solver.py

# Third-order model
python sympnets/scripts/3order_model_solver.py
```
