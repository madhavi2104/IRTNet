import pandas as pd
import matplotlib.pyplot as plt

# Visualize Results from a Specified CSV File
def visualize_results_from_csv(csv_path):
    print(f"\nProcessing file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Ensure all column names are consistent
    if "True Label" not in df.columns:
        print("Error: 'True Label' column is missing in the CSV file.")
        return
    
    # Extract model names from the columns (excluding "Item" and "True Label")
    model_columns = [col for col in df.columns if col.startswith("Answer")]
    true_label_column = "True Label"

    # Normalize labels for comparison (strip spaces, lower case)
    df[true_label_column] = df[true_label_column].str.strip().str.lower()
    for model in model_columns:
        df[model] = df[model].str.strip().str.lower()

    # Debugging: Print first few rows to verify normalization
    print("\nSample rows after normalization:")
    print(df.head())

    model_performance = {}
    for model in model_columns:
        correct = df[df[model] == df[true_label_column]]
        incorrect = df[df[model] != df[true_label_column]]
        model_performance[model] = {"Correct": len(correct), "Incorrect": len(incorrect)}

    # Generate comparison chart for all models
    model_names = list(model_performance.keys())
    correct_counts = [model_performance[model]["Correct"] for model in model_names]
    incorrect_counts = [model_performance[model]["Incorrect"] for model in model_names]

    x = range(len(model_names))
    plt.figure(figsize=(10, 6))
    plt.bar(x, correct_counts, width=0.4, label="Correct", color="green", align="center")
    plt.bar(x, incorrect_counts, width=0.4, label="Incorrect", color="red", align="edge")
    plt.xticks(x, model_names, rotation=45)
    plt.xlabel("Models")
    plt.ylabel("Number of Images")
    plt.title("Comparison of Model Predictions")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main Function
def main():
    # Specify the path to the desired CSV file
    csv_path = "data/Processed/ImageNet-V2_corrected_results.csv"  # Update this path to your desired CSV file

    if not csv_path or not csv_path.endswith(".csv"):
        print("Error: Please specify a valid CSV file path.")
        return

    visualize_results_from_csv(csv_path)

if __name__ == "__main__":
    main()

