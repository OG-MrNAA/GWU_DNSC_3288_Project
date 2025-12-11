
# Predict Future Sales – Model Card (DNSC 3288, Fall 2025)

## 1. Basic Information

- **Course**: DNSC 3288 (Machine Learning)
- **Project Title**: Predict Future Sales (Kaggle Competition)
- **Student(s)**: Naim Abu Altayeb (naimabualtayeb@gwu.edu)
- **Date**: 2025-12-10
- **Model Version**: 1.0
- **License**: MIT
- **Implementation**: [Jupyter Notebook](./DNSC_3288_Predict_future_sales_Project.ipynb)

### Reference Examples

We referred to the Kaggle notebook “Predict Future Sales - Arima - Prophet - XGBOOST” and the GitHub repository `GWU_DNSC_6301_project` for ideas about workflow and documentation style. All final modeling and decisions in this project are our own.

### Intended Use

- **Intended Uses**: Educational demonstration of time-series forecasting using regression trees; example for DNSC 3288 project.
- **Intended Users**: Instructors, students, and data science beginners.
- **Out-of-Scope Uses**: High-stakes financial planning, inventory management for real-world retail without further validation.

## 2. Training Data

- **Source**: Kaggle 'Predict Future Sales' competition dataset (`sales_train.csv`).
- **Split Method**: Time-based split.
  - **Training**: First 32 months (date_block_num <= 31).
  - **Validation**: Subsequent 2 months (date_block_num 32, 33).
- **Size**:
  - Training Rows: 675441
  - Validation Rows: 35700

### Processing Steps (Modeling Table)

- Aggregated daily `item_cnt_day` to `item_cnt_month` for each `(date_block_num, shop_id, item_id)`.
- Clipped `item_cnt_month` to the range [0, 20] to follow the competition guidelines.
- Created lag features (`item_cnt_lag_1`, `item_cnt_lag_2`, `item_cnt_lag_3`) for each `(shop_id, item_id)` based on previous months.

### Data Dictionary (Features Used)

| Column Name | Role | Level | Description |
|---|---|---|---|
| `shop_id` | Feature | Nominal | Unique identifier for a shop. |
| `item_id` | Feature | Nominal | Unique identifier for a product. |
| `item_category_id` | Feature | Nominal | Category identifier for the product. |
| `item_price_avg` | Feature | Continuous | Average price of the item for the month/shop. |
| `item_cnt_lag_1` | Feature | Continuous | Sales count of the item 1 month ago. |
| `item_cnt_lag_2` | Feature | Continuous | Sales count of the item 2 months ago. |
| `item_cnt_lag_3` | Feature | Continuous | Sales count of the item 3 months ago. |
| `date_block_num` | Feature | Ordinal | Consecutive month number (0 to 33). |
| `item_cnt_month` | Target | Continuous | Monthly sales count (clipped to [0, 20]). |

## 3. Test Data

- **Source**: Kaggle `test.csv`.
- **Rows**: 214200
- **Differences**: Contains only `ID`, `shop_id`, and `item_id` for month 34. Target `item_cnt_month` is unknown.

## 4. Model Details

- **Type**: XGBoost Regressor (Gradient Boosting)
- **Software**:
  - Python 3, 3.12.12
  - scikit-learn, 1.6.1
  - xgboost, 3.1.2
- **Hyperparameters**:
  - `max_depth`: 8
  - `n_estimators`: 50
  - `learning_rate`: 0.1
  - `objective`: reg:squarederror

## 5. Quantitative Analysis

We evaluated the model using **RMSE** (standard regression metric), **AUC** (Area Under Curve for binary 'sales > 0' task), and **AIR** (Adverse Impact Ratio across item categories for the binary task).

| Dataset | RMSE | AUC (Binary) | AIR (Fairness) |
|---|---|---|---|
| **Training** | 1.5090 | 0.5580 | 1.0000 |
| **Validation** | 1.4736 | 0.5189 | 1.0000 |
| **Test (Kaggle public/private)** | Public: 1.50735, Private: 1.50747 | N/A | N/A |

*Note: Test AUC and AIR are N/A because ground truth labels are held by Kaggle. Only RMSE is available via the Kaggle public and private leaderboards.*

### Plots
- **Figure 1**: Monthly sales trends show seasonality and overall trend (saved as `figures/monthly_sales.png`).
- **Figure 2**: Feature importance plot confirms that `item_cnt_lag_1` (previous month's sales) is the most predictive feature (saved as `figures/feature_importance.png`).

## 6. Ethical Considerations

- **Math / Software Issues**
  - **Overfitting**: The model relies heavily on lag features from past months. If market dynamics change (new products, store closings, promotions), these patterns may no longer hold.
  - **Data Leakage**: We used a strict time-based split on `date_block_num` to avoid using future information, but coding mistakes could still introduce leakage in a more complex version of the model.

- **Real-World Risks**
  - **Stockouts / Overstock**: In a real retail setting, under-prediction could cause stockouts and lost sales, while over-prediction could lead to excess inventory and waste.
  - **Resource Allocation**: If some shops or categories are systematically under-predicted, they might receive fewer resources or less attention.

- **Uncertainties**
  - The data comes from one retailer in one country over 2013–2015. It is unclear how well this model would generalize to other years, other retailers, or other markets.
  - We did not include many external factors (holidays, promotions, macroeconomic conditions), so it is uncertain how sensitive the model is to these missing variables.
  - Due to time and compute limits, we could not explore many alternative model classes or feature sets.

- **Unexpected Results**
  - The AUC values for the binary “sold at least one unit” task are only slightly above 0.5, which suggests that this classification view of the task is quite hard and many items rarely sell.
  - The AIR metric is 1.0 across the broad item-category groups we tested. This likely reflects very low positive rates across all categories rather than strong evidence of true fairness. A production system would need a deeper bias and subgroup analysis before deployment.
