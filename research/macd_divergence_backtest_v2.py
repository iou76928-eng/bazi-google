import pandas as pd

# The original script uses bars.hist as shorthand for the 'hist' column.
# pandas reserves DataFrame.hist as a plotting method, so expose the column
# through a property for this isolated backtest run.
pd.DataFrame.hist = property(lambda frame: frame["hist"])

import macd_divergence_backtest as backtest

if __name__ == "__main__":
    backtest.main()
