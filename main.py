import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from tkinter import *
import tkinter as tk
from tkinter import ttk
import joblib

data = pd.read_csv("house_data.csv")
# print(data["price"].describe())

### --- EAD ---###

# print(data.head())
# print(data.info())
# print(data.describe())

# print(data.nunique())

# unique_counts = data.nunique()

# plt.figure(figsize=(12, 6))
# unique_counts.plot(kind="bar", color="skyblue", edgecolor="black")

# plt.title("unique counts bar", fontsize=14)
# plt.xlabel("", fontsize=12)
# plt.ylabel("counts", fontsize=12)
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

### --- Created Dataset ---###


def preProcess(data):

    processed_file = "processed_mainData.csv"
    if os.path.exists(processed_file):
        print("Loading processed data from CSV...")
        mainData = pd.read_csv(processed_file)
    else:
        data["date"] = pd.to_datetime(data["date"])

        mainData = pd.DataFrame()

        mainData["sale_year"] = data["date"].dt.year
        mainData["sale_month"] = data["date"].dt.month
        mainData["sale_season"] = data["date"].dt.quarter

        mainData["Bedrooms"] = data.loc[data["bedrooms"] != 0, "bedrooms"]
        mainData["Bathrooms"] = data.loc[data["bathrooms"] != 0, "bathrooms"]
        # print(mainData[mainData["Bedrooms"] == 0].shape[0])
        # print(mainData[mainData["Bathrooms"] == 0].shape[0])

        mainData["LivingArea"] = data["sqft_living"]
        mainData["LandArea"] = data["sqft_lot"]
        mainData["Floor"] = data["floors"]
        mainData["Waterfront"] = data["waterfront"]
        mainData["View"] = data["view"]
        mainData["Condition"] = data["condition"]
        mainData["Grade"] = data["grade"]
        mainData["AboveArea"] = data["sqft_above"]
        mainData["BasementArea"] = data["sqft_basement"]
        mainData["YearBuilt"] = data["yr_built"]
        mainData["YearRenovated"] = data["yr_renovated"]
        mainData["AreaCode"] = data["zipcode"]
        mainData["XCoord"] = data["long"]
        mainData["YCoord"] = data["lat"]
        mainData["Neighbor_LivingArea_Avg"] = data["sqft_living15"]
        mainData["Neighbor_LandArea_Avg"] = data["sqft_lot15"]

        mainData["Larger_Than_Neighbors"] = (
            (data["sqft_living"] > data["sqft_living15"]) & (data["sqft_lot"] > data["sqft_lot15"])
        ).astype(int)

        mainData["Price"] = data["price"]

        mainData.to_csv(processed_file, index=False, encoding="utf-8-sig")


# preProcess(data)

mainData = pd.read_csv("processed_mainData.csv")


def modeling(mainData):

    X = mainData.drop(columns=["Price"])
    y = mainData["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # model = xgb.XGBRegressor(
    #     objective="reg:squarederror",
    #     n_estimators=100,
    #     random_state=42,
    #     colsample_bytree=0.5,
    #     learning_rate=0.1,
    #     max_depth=6,
    #     subsample=0.8,
    #     min_child_weight=7,
    #     max_delta_step=0,
    #     max_bin=128,
    #     gamma=0.1,
    # )
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # r2 = r2_score(y_test, y_pred)

    # print(f"RMSE: {rmse:.3f}")
    # print(f"R²: {r2:.3f}")

    # model = xgb.XGBRegressor()
    # param_grid = {
    #     "max_bin": [128, 256, 512, 1024],
    #     "gamma": [0, 0.1, 0.5, 1, 5],
    #     "max_depth": [3, 4, 5, 6, 8, 10],
    #     "min_child_weight": [1, 3, 5, 7, 10],
    #     "learning_rate": np.arange(0.1, 0.5, 0.1),
    #     "max_delta_step": [0, 1, 5, 10],
    #     "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    #     "colsample_bytree": [0.8, 1.0],
    #     "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    # }

    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)
    # search = RandomizedSearchCV(
    #     estimator=model,
    #     param_distributions=param_grid,
    #     n_iter=100,
    #     cv=5,
    #     scoring="neg_root_mean_squared_error",
    #     n_jobs=-1,
    #     verbose=2,
    #     random_state=42,
    # )
    # search.fit(X_train, y_train)

    # print("Best parameters:", search.best_params_)

    # model = xgb.XGBRegressor()
    # param_grid = {
    #     "max_depth": [3, 6, 9],
    #     "learning_rate": [0.01, 0.1, 0.2],
    #     "subsample": [0.8, 1.0],
    #     "colsample_bytree": [0.8, 1.0],
    # }

    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    # grid_search.fit(X_train, y_train)

    # print("Best parameters:", grid_search.best_params_)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=6,
        learning_rate=0.1,
        subsample=1,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # r2 = r2_score(y_test, y_pred)

    # print(f"RMSE: {rmse:.3f}")
    # print(f"R²: {r2:.3f}")

    return model


# model = modeling(mainData)
# joblib.dump(model, "house_price_model.pkl")

model = joblib.load("house_price_model.pkl")


## model testing
# X = mainData.drop(columns=["Price"])
# y = mainData["Price"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# y_pred = model.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print(f"RMSE: {rmse:.3f}")
# print(f"R²: {r2:.3f}")


year_entry = None
combo1 = None
bath_entry = None
area_entry = None
sakht_entry = None
tabaghat_combo = None
extra_entry1 = None
extra_widget_added = [False]
v2 = None
entry4_1 = None
entry4_2 = None
entry10_1 = None
entry10_2 = None
entry17 = None
entry18 = None


def predict_price():
    sale_year = int(year_entry.get())
    sale_month = int(combo1.get())
    sale_season = (sale_month - 1) // 3 + 1
    bedrooms = int(entry_bedrooms.get())
    bathrooms = float(hamam_entry.get())
    living_area = float(area_entry.get())
    land_area = float(sakht_entry.get())
    floor = int(tabaghat_combo.get())
    waterfront = int(v2.get())
    view = int(combo2.get().split()[0])
    condition = int(combo3.get().split()[0])
    grade = int(combo4.get().split()[0])
    above_area = float(extra_entry1.get()) if extra_widget_added[0] else 0
    basement_area = float(extra_entry2[0].get()) if extra_widget_added2[0] else 0
    year_built = int(entry4_1.get())
    year_renovated = int(extra_entry4_2[0].get()) if extra_widget_added4_2[0] else 0
    area_code = int(entry4_2.get())
    xcoord = float(entry10_2.get())
    ycoord = float(entry10_1.get())
    neighbor_living = float(entry17.get())
    neighbor_land = float(entry18.get())
    larger_than_neighbors = int((living_area > neighbor_living) and (land_area > neighbor_land))

    features = [
        [
            sale_year,
            sale_month,
            sale_season,
            bedrooms,
            bathrooms,
            living_area,
            land_area,
            floor,
            waterfront,
            view,
            condition,
            grade,
            above_area,
            basement_area,
            year_built,
            year_renovated,
            area_code,
            xcoord,
            ycoord,
            neighbor_living,
            neighbor_land,
            larger_than_neighbors,
        ]
    ]

    price = model.predict(features)[0]

    result_label.config(text=f"قیمت پیش‌بینی شده: {price:,.0f} دلار")


def GUI():

    global year_entry, combo1, combo2, combo3, combo4, bath_entry, area_entry, sakht_entry, tabaghat_combo
    global extra_entry1, extra_widget_added, v2, entry4_1, entry4_2
    global entry10_1, entry10_2, entry17, entry18
    global entry_bedrooms, hamam_entry
    global extra_entry2, extra_widget_added2
    global extra_entry4_2, extra_widget_added4_2
    global v, v4_2
    global result_label

    window = tk.Tk()
    window.title("محاسبه قیمت خانه")
    window.geometry("600x720")

    window.grid_columnconfigure(0, minsize=300)
    window.grid_columnconfigure(1, minsize=300)

    # row 0
    label1 = tk.Label(window, text="اطلاعات مربوط به زمان فروش", font=("Arial", 14))
    label1.grid(row=0, column=1, sticky="e", padx=(0, 20), pady=(10, 0))

    # row 1
    label2 = tk.Label(window, text="سال فروش", font=("Arial", 12))
    label2.grid(row=1, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    year_entry = tk.Entry(window)
    year_entry.grid(row=1, column=1, sticky="w", padx=(40, 0), pady=(10, 0))

    label3 = tk.Label(window, text="ماه فروش", font=("Arial", 12))
    label3.grid(row=1, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
    months = [str(i) for i in range(1, 13)]
    combo1 = ttk.Combobox(window, values=months, state="readonly", width=17)
    combo1.current(0)
    combo1.grid(row=1, column=0, sticky="w", padx=(40, 0), pady=(10, 0))

    # row 3
    label4 = tk.Label(window, text="اطلاعات مربوط به ویژگی های خانه", font=("Arial", 14))
    label4.grid(row=3, column=1, sticky="e", padx=(0, 20), pady=(20, 0))

    # row 4
    label4_1 = tk.Label(window, text="سال ساخت", font=("Arial", 12))
    label4_1.grid(row=4, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    entry4_1 = tk.Entry(window)
    entry4_1.grid(row=4, column=1, sticky="w", padx=(40, 0), pady=(10, 0))

    label4_2 = tk.Label(window, text="کد پستی منطقه", font=("Arial", 12))
    label4_2.grid(row=4, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
    entry4_2 = tk.Entry(window)
    entry4_2.grid(row=4, column=0, sticky="w", padx=(40, 0), pady=(10, 0))

    # row 5
    label5 = tk.Label(window, text="متراژ زیر بنا", font=("Arial", 12))
    label5.grid(row=5, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
    area_entry = tk.Entry(window)
    area_entry.grid(row=5, column=0, sticky="w", padx=(40, 0), pady=(10, 0))

    label6 = tk.Label(window, text="تعداد اتاق خواب", font=("Arial", 12))
    label6.grid(row=5, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    entry_bedrooms = tk.Entry(window)
    entry_bedrooms.grid(row=5, column=1, sticky="w", padx=(40, 0), pady=(10, 0))

    # row 6
    label6 = tk.Label(window, text="تعداد حمام", font=("Arial", 12))
    label6.grid(row=6, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    hamam_entry = tk.Entry(window)
    hamam_entry.grid(row=6, column=1, sticky="w", padx=(40, 0), pady=(10, 0))

    label7 = tk.Label(window, text="متراژ ساخت", font=("Arial", 12))
    label7.grid(row=6, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
    sakht_entry = tk.Entry(window)
    sakht_entry.grid(row=6, column=0, sticky="w", padx=(40, 0), pady=(10, 0))

    # row 7
    label8 = tk.Label(window, text="تعداد طبقات", font=("Arial", 12))
    label8.grid(row=7, column=1, sticky="e", padx=(0, 40), pady=(10, 0))

    tabaghat_options = ["1", "2", "3", "4"]
    tabaghat_combo = ttk.Combobox(window, values=tabaghat_options, state="readonly", width=17)
    tabaghat_combo.current(0)
    tabaghat_combo.grid(row=7, column=1, sticky="w", padx=(40, 0), pady=(10, 0))

    label_extra1 = tk.Label(window, text="متراژ بخش بالای خانه", font=("Arial", 12))
    extra_entry1 = tk.Entry(window, width=15)

    extra_widget_added = [False]

    def check_tabaghat(event):
        value = int(tabaghat_combo.get())

        if value > 1 and not extra_widget_added[0]:
            label_extra1.grid(row=7, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
            extra_entry1.grid(row=7, column=0, sticky="w", padx=(40, 0), pady=(10, 0))
            extra_widget_added[0] = True

        elif value == 1 and extra_widget_added[0]:
            label_extra1.grid_remove()
            extra_entry1.grid_remove()
            extra_widget_added[0] = False

    tabaghat_combo.bind("<<ComboboxSelected>>", check_tabaghat)

    # row 8
    label9 = tk.Label(window, text="دارای زیر زمین", font=("Arial", 12))
    label9.grid(row=8, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    v = tk.IntVar(value=2)
    radio_frame = tk.Frame(window)
    radio_frame.grid(row=8, column=1, sticky="w", padx=(30, 0), pady=(10, 0))

    extra_widget_added2 = [False]
    label_extra2 = [None]
    extra_entry2 = [None]

    def check_zirzamin():
        try:
            value = int(v.get())
            if value == 1 and not extra_widget_added2[0]:
                label_extra2[0] = tk.Label(window, text="متراژ زیر زمین", font=("Arial", 12))
                label_extra2[0].grid(row=8, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
                extra_entry2[0] = tk.Entry(window, width=15)
                extra_entry2[0].grid(row=8, column=0, sticky="w", padx=(40, 0), pady=(10, 0))
                extra_widget_added2[0] = True
            elif value == 2 and extra_widget_added2[0]:
                if label_extra2[0]:
                    label_extra2[0].grid_remove()
                if extra_entry2[0]:
                    extra_entry2[0].grid_remove()
                extra_widget_added2[0] = False
        except ValueError:
            pass

    tk.Radiobutton(radio_frame, text="بله", variable=v, value=1, command=check_zirzamin).pack(side="left", padx=5)
    tk.Radiobutton(radio_frame, text="خیر", variable=v, value=2, command=check_zirzamin).pack(side="left", padx=5)

    # row 9
    label4_2 = tk.Label(window, text="دارای نوسازی", font=("Arial", 12))
    label4_2.grid(row=9, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    v4_2 = tk.IntVar(value=2)
    radio_frame = tk.Frame(window)
    radio_frame.grid(row=9, column=1, sticky="w", padx=(30, 0), pady=(10, 0))

    extra_widget_added4_2 = [False]
    label_extra4_2 = [None]
    extra_entry4_2 = [None]

    def check_bazsaazi():
        try:
            value = int(v4_2.get())
            if value == 1 and not extra_widget_added4_2[0]:
                label_extra4_2[0] = tk.Label(window, text="سال بازسازی", font=("Arial", 12))
                label_extra4_2[0].grid(row=9, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
                extra_entry4_2[0] = tk.Entry(window, width=15)
                extra_entry4_2[0].grid(row=9, column=0, sticky="w", padx=(40, 0), pady=(10, 0))
                extra_widget_added4_2[0] = True
            elif value == 2 and extra_widget_added4_2[0]:
                if label_extra4_2[0]:
                    label_extra4_2[0].grid_remove()
                if extra_entry4_2[0]:
                    extra_entry4_2[0].grid_remove()
                extra_widget_added4_2[0] = False
        except ValueError:
            pass

    tk.Radiobutton(radio_frame, text="بله", variable=v4_2, value=1, command=check_bazsaazi).pack(side="left", padx=5)
    tk.Radiobutton(radio_frame, text="خیر", variable=v4_2, value=2, command=check_bazsaazi).pack(side="left", padx=5)

    # row 10
    label10_1 = tk.Label(window, text="عرض جغرافیایی", font=("Arial", 12))
    label10_1.grid(row=10, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    entry10_1 = tk.Entry(window)
    entry10_1.grid(row=10, column=1, sticky="w", padx=(40, 0), pady=(10, 0))

    label10_2 = tk.Label(window, text="طول جغرافیایی", font=("Arial", 12))
    label10_2.grid(row=10, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
    entry10_2 = tk.Entry(window)
    entry10_2.grid(row=10, column=0, sticky="w", padx=(40, 0), pady=(10, 0))

    # row 11
    label11 = tk.Label(window, text="ویژگی های اضافه", font=("Arial", 14))
    label11.grid(row=11, column=1, sticky="e", padx=(0, 20), pady=(20, 0))

    # row 12
    label12 = tk.Label(window, text="ایا مشرف به آب است ؟", font=("Arial", 12))
    label12.grid(row=12, column=1, sticky="e", padx=(0, 40), pady=(10, 0))

    v2 = tk.IntVar(value=2)
    radio_frame2 = tk.Frame(window)
    radio_frame2.grid(row=12, column=1, sticky="w", padx=(30, 0), pady=(10, 0))
    tk.Radiobutton(radio_frame2, text="بله", variable=v2, value=1).pack(side="left", padx=5)
    tk.Radiobutton(radio_frame2, text="خیر", variable=v2, value=2).pack(side="left", padx=5)

    label13 = tk.Label(window, text="میزان دید و چشم انداز", font=("Arial", 12))
    label13.grid(row=12, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
    combo2 = ttk.Combobox(window, values=["0 (کمترین)", "1", "2", "3", "4 (بیشترین)"], state="readonly", width=10)
    combo2.current(0)
    combo2.grid(row=12, column=0, sticky="w", padx=(40, 0), pady=(10, 0))

    # row 13
    label14 = tk.Label(window, text="وضعیت کلی سلامت خانه", font=("Arial", 12))
    label14.grid(row=13, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    combo3 = ttk.Combobox(window, values=["1 (کمترین)", "1", "2", "3", "4", "5 (بیشترین)"], state="readonly", width=10)
    combo3.current(0)
    combo3.grid(row=13, column=1, sticky="w", padx=(40, 0), pady=(10, 0))

    label14 = tk.Label(window, text="کیفیت ساخت و طراحی", font=("Arial", 12))
    label14.grid(row=13, column=0, sticky="e", padx=(0, 40), pady=(10, 0))
    combo4 = ttk.Combobox(
        window,
        values=["1 (کمترین)", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13 (بیشترین)"],
        state="readonly",
        width=10,
    )
    combo4.current(0)
    combo4.grid(row=13, column=0, sticky="w", padx=(40, 0), pady=(10, 0))

    # row 14
    label15 = tk.Label(window, text="خانه مد نظر در مقایسه با خانه های اطراف", font=("Arial", 14))
    label15.grid(row=14, column=1, sticky="e", padx=(0, 20), pady=(20, 0))

    # row 15
    label16 = tk.Label(window, text="میانگین مساحت زیربنای 15 خانه اطراف", font=("Arial", 12))
    label16.grid(row=15, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    entry17 = tk.Entry(window)
    entry17.grid(row=15, column=0, sticky="w", padx=(170, 0), pady=(10, 0))

    # row 16
    label17 = tk.Label(window, text="میانگین مساحت کلی زمین 15 خانه اطراف", font=("Arial", 12))
    label17.grid(row=16, column=1, sticky="e", padx=(0, 40), pady=(10, 0))
    entry18 = tk.Entry(window)
    entry18.grid(row=16, column=0, sticky="w", padx=(170, 0), pady=(10, 0))

    # row 17
    button = tk.Button(window, text="محاسبه قیمت", width=25, command=predict_price)
    button.grid(row=17, column=0, columnspan=2, sticky="n", pady=(30, 0))

    # row 18
    result_label = tk.Label(window, text="", font=("Arial", 14))
    result_label.grid(row=18, column=0, columnspan=2, pady=(20, 0))

    window.mainloop()


GUI()
