---
license: odbl
---

# 🏠 Divar Real Estate Ads Dataset

[![Dataset Size](https://img.shields.io/badge/Size-750%20MB-blue)](https://huggingface.co/datasets/divar/real-estate-ads)
[![Rows](https://img.shields.io/badge/Rows-1M-green)](https://huggingface.co/datasets/divar/real-estate-ads)

## 📋 Overview

The `real_estate_ads` dataset contains one million anonymized real estate advertisements collected from the [Divar](https://divar.ir) platform, one of the largest classified ads platforms in the Middle East. This comprehensive dataset provides researchers, data scientists, and entrepreneurs with authentic real estate market data to build innovative solutions such as price evaluation models, market analysis tools, and forecasting systems.


## 🔍 Dataset Details

| Property        | Value                                      |
| --------------- | ------------------------------------------ |
| **Size**        | 1,000,000 rows, approximately 750 MB       |
| **Time Period** | Six-month period (2024)               |
| **Source**      | Anonymized real estate listings from Divar |
| **Format**      | Tabular data (CSV/Parquet) with 57 columns |
| **Languages**   | Mixed (primarily Persian)                  |
| **Domains**     | Real Estate, Property Market               |

## 🚀 Quick Start

```python
# Load the dataset using the Hugging Face datasets library
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("divarofficial/real-estate-ads")

# Print the first few examples
print(dataset['train'][:5])

# Get dataset statistics
print(f"Dataset size: {len(dataset['train'])} rows")
print(f"Features: {dataset['train'].features}")
```

## 📊 Schema

The dataset includes comprehensive property information organized in the following categories:

### 🏷️ Categorization

- `cat2_slug`, `cat3_slug`: Property categorization slugs
- `property_type`: Type of property (apartment, villa, land, etc.)

### 📍 Location

- `city_slug`, `neighborhood_slug`: Location identifiers
- `location_latitude`, `location_longitude`: Geographic coordinates
- `location_radius`: Location accuracy radius

### 📝 Listing Details

- `created_at_month`: Timestamp of when the ad was created
- `user_type`: Type of user who posted the listing (individual, agency, etc.)
- `description`, `title`: Textual information about the property

### 💰 Financial Information

- **Rent-related**: `rent_mode`, `rent_value`, `rent_to_single`, `rent_type`
- **Price-related**: `price_mode`, `price_value`
- **Credit-related**: `credit_mode`, `credit_value`
- **Transformed values**: Various transformed financial metrics for analysis

### 🏢 Property Specifications

- `land_size`, `building_size`: Property dimensions (in square meters)
- `deed_type`, `has_business_deed`: Legal property information
- `floor`, `rooms_count`, `total_floors_count`, `unit_per_floor`: Building structure details
- `construction_year`, `is_rebuilt`: Age and renovation status

### 🛋️ Amenities and Features

- **Utilities**: `has_water`, `has_electricity`, `has_gas`
- **Climate control**: `has_heating_system`, `has_cooling_system`
- **Facilities**: `has_balcony`, `has_elevator`, `has_warehouse`, `has_parking`
- **Luxury features**: `has_pool`, `has_jacuzzi`, `has_sauna`
- **Other features**: `has_security_guard`, `has_barbecue`, `building_direction`, `floor_material`

### 🏨 Short-term Rental Information

- `regular_person_capacity`, `extra_person_capacity`
- `cost_per_extra_person`
- **Pricing variations**: `rent_price_on_regular_days`, `rent_price_on_special_days`, `rent_price_at_weekends`

## 📈 Example Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert to pandas DataFrame for analysis
df = dataset['train'].to_pandas()

# Price distribution by property type
plt.figure(figsize=(12, 6))
sns.boxplot(x='property_type', y='price_value', data=df)
plt.title('Price Distribution by Property Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation between building size and price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='building_size', y='price_value', data=df)
plt.title('Correlation between Building Size and Price')
plt.xlabel('Building Size (sq.m)')
plt.ylabel('Price')
plt.tight_layout()
plt.show()
```

## 💡 Use Cases

This dataset is particularly valuable for:

1. **Price Prediction Models**: Train algorithms to estimate property values based on features

   ```python
   # Example: Simple price prediction model
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.model_selection import train_test_split

   features = ['building_size', 'rooms_count', 'construction_year', 'has_parking']
   X = df[features].fillna(0)
   y = df['price_value'].fillna(0)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   model = RandomForestRegressor(n_estimators=100)
   model.fit(X_train, y_train)
   ```

2. **Market Analysis**: Understand trends and patterns in the real estate market
3. **Recommendation Systems**: Build tools to suggest properties based on user preferences
4. **Natural Language Processing**: Analyze property descriptions and titles
5. **Geospatial Analysis**: Study location-based pricing and property distribution

## 🔧 Data Processing Information

The data has been:

- Anonymized to protect privacy
- Randomly sampled from the complete Divar platform dataset
- Cleaned with select columns removed to ensure privacy and usability
- Standardized to ensure consistency across entries

## 📚 Citation and Usage

When using this dataset in your research or applications, please consider acknowledging the source:

```bibtex
@dataset{divar2025realestate,
  author = {Divar Corporation},
  title = {Real Estate Ads Dataset from Divar Platform},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/divar/real-estate-ads}
}
```
## 🤝 Contributing

We welcome contributions to improve this dataset! If you find issues or have suggestions, please open an issue on the [GitHub repository](https://github.com/divar-ir/kenar-docs) or contact us at [kenar.support@divar.ir](mailto:kenar.support@divar.ir).