# Hotel Booking Cancellations: Data Science Capstone

Week 6 capstone for the YuvaIntern Data Science with Python internship.

The project cleans the public Hotel Booking Demand dataset, explores it, segments guests with K-Means, and predicts cancellations with Logistic Regression and a Random Forest.

## Run it

1. Download the data: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv (save as `hotels.csv` next to the script)
2. Install packages: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Run: `python capstone.py`

The script prints all results and saves the charts to a `figs` folder.

## Files

- `capstone.py`: full pipeline (cleaning, EDA, clustering, modelling)
- `figs/`: charts used in the report
- `Week6_Capstone_Hotel_Booking_Report.docx`: the report

## Main results

- 119,390 raw bookings cleaned to 87,224; cancellation rate 27.5%
- 4 guest segments (silhouette 0.22)
- Random Forest test ROC-AUC 0.851 (logistic regression 0.801)
