import matplotlib.pyplot as plt
import pandas as pd


def plot_price_with_sma(
    df: pd.DataFrame,
    ticker: str,
    sma_short: int,
    sma_long: int,
    save_path: str = "chart.png",
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["Close"], label="Cena zamknięcia", linewidth=1.5)
    ax.plot(df.index, df[f"SMA_{sma_short}"], label=f"SMA {sma_short}", linewidth=1.2)
    ax.plot(df.index, df[f"SMA_{sma_long}"], label=f"SMA {sma_long}", linewidth=1.2)

    ax.set_title(f"{ticker} — cena i średnie kroczące")
    ax.set_xlabel("Data")
    ax.set_ylabel("Cena (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Wykres zapisany: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
