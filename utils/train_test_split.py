from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == '__main__':
    dataset = pd.read_csv('../data/annotation.csv')
    train_df, val_df = train_test_split(
        dataset['file_name'].unique(),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_df = dataset[dataset['file_name'].isin(train_df)].to_csv('../data/annotation_train.csv')
    val_df = dataset[dataset['file_name'].isin(val_df)].to_csv('../data/annotation_val.csv')