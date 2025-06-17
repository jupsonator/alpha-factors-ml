import os
import pandas as pd

print("Running compute_labels.py")

def compute_forward_return(df, days=21):
    df = df.sort_index()
    df['forward_return'] = df['Close'].shift(-days) / df['Close'] - 1
    return df.dropna(subset=['forward_return'])

def main(input_dir="data/features", output_dir="data/labelled"):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
            df = compute_forward_return(df)
            base_name = filename.split("_")[0]  
            output_filename = f"{base_name}_labelled.csv"
            df.to_csv(os.path.join(output_dir, output_filename))
            print(f"Labelled: {output_filename}")

if __name__ == "__main__":
    main()