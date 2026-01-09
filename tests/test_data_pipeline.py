def test_clean_data_removes_missing_and_duplicates(sample_df):
    from src.data_pipeline import clean_data
    cleaned = clean_data(sample_df)
    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.duplicated().sum() == 0

def test_validate_data_schema(cleaned_df):
    from src.data_pipeline import validate_data
    validated = validate_data(cleaned_df)
    assert not validated.empty
