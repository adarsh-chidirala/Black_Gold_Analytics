**Pseudocode Plan:**
1. Title and project description.
2. Prerequisites and installation instructions.
3. Usage instructions (how to run the script).
4. Output description.
5. Example command.
6. License section.

---

```markdown
# Oil Market EDA

This project provides exploratory data analysis (EDA) for global oil production and consumption trends using Python. It generates visualizations from a cleaned dataset.

## Prerequisites

- Python 3.8+
- pandas
- matplotlib
- seaborn

Install dependencies with:
```bash
pip install pandas matplotlib seaborn
```

## Usage

1. Place your cleaned dataset at `data_clean/master_oil_panel.csv`.
2. Run the EDA script:
   ```bash
   python generate_eda.py
   ```
3. Output charts will be saved in the `outputs/` directory.

## Output

- Visualizations of global oil production and consumption trends (1980-2023).

#