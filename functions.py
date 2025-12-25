from __future__ import annotations
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack, issparse

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

GENDER_MAP = {1: "Female", 2: "Male"}

EXPERIENCE_MAP = {
    1: "Beginner",
    2: "Intermediate",
    3: "Advanced",
}

DIET_MAP = {
    1: "Balanced",
    2: "Keto",
    3: "Low-Carb",
    4: "Paleo",
    5: "Vegan",
    6: "Vegetarian",
}

# What the user sees in the menu (friendly labels)
GOAL_MENU = {
    1: "Lose weight",
    2: "Maintain weight",
    3: "Gain weight",
}

# What gets stored in the data (canonical values)
GOAL_MAP = {
    1: "Loss",
    2: "Maintain",
    3: "Gain",
}


def _choose_from_menu(title: str, options: dict[int, str]) -> int:
    """Print a menu and return the selected option key (int)."""
    while True:
        print(f"\n{title}")
        for k, v in options.items():
            print(f"  {k} - {v}")
        try:
            choice = int(input("Enter choice: ").strip())
            if choice in options:
                return choice
            print("Invalid choice. Please enter one of the numbers shown.")
        except ValueError:
            print("Please enter a number.")


def _ask_int(prompt: str, min_val: int | None = None, max_val: int | None = None) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if min_val is not None and val < min_val:
                print(f"Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("Please enter an integer.")


def _ask_float(prompt: str, min_val: float | None = None, max_val: float | None = None) -> float:
    while True:
        try:
            val = float(input(prompt).strip())
            if min_val is not None and val < min_val:
                print(f"Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("Please enter a numeric value.")


def _print_summary(user_data: dict) -> None:
    print("\n==============================")
    print("INPUT SUMMARY")
    print("==============================")
    for k, v in user_data.items():
        if isinstance(v, float):
            print(f"{k:35s}: {v:.2f}")
        else:
            print(f"{k:35s}: {v}")
    print("==============================\n")


def collect_new_user() -> dict:
    print("\n==============================")
    print("NEW USER INPUT FORM")
    print("==============================")

    gender_choice = _choose_from_menu("Gender", GENDER_MAP)
    exp_choice = _choose_from_menu("Experience Level", EXPERIENCE_MAP)
    diet_choice = _choose_from_menu("Diet Type", DIET_MAP)
    goal_choice = _choose_from_menu("Goal", GOAL_MENU)

    age = _ask_int("\nAge: ", min_val=5, max_val=120)
    weight_kg = _ask_float("Weight (kg): ", min_val=20, max_val=400)
    height_m = _ask_float("Height (m): ", min_val=0.8, max_val=2.5)

    workout_freq = _ask_int("Workout Frequency (days): ", min_val=0, max_val=7)
    meals_freq = _ask_int("Daily meals frequency: ", min_val=1, max_val=10)
    if goal_choice == 2:
        weight_change = 0
    else:
        weight_change = _ask_float(
            "WeightChange (kg) (enter 5 if you want to lose/gain 5 kg): ",
            min_val=-200,
            max_val=200,
        )
    goal_days = _ask_int("GoalDays (days): ", min_val=1, max_val=3650)

    bmi = weight_kg / (height_m ** 2)

    user_data = {
        "Age": age,
        "Gender": GENDER_MAP[gender_choice],
        "Weight (kg)": weight_kg,
        "Height (m)": height_m,
        "BMI": bmi,
        "Experience_Level": exp_choice,
        "Workout_Frequency (days)": workout_freq,
        "Daily meals frequency": meals_freq,
        "diet_type": DIET_MAP[diet_choice],

        "Goal": GOAL_MAP[goal_choice],
        "WeightChange (kg)": weight_change,
        "GoalDays": goal_days,
    }

    _print_summary(user_data)
    return user_data


# Predicts cluster_id for the dataframe and adds the column cluster_id to dataframe
# This same function is used for the new_user dataframe containing only one row of new user

def predict_cluster_id(
    user_df: pd.DataFrame,
    artifacts_path: str = "models/kmeans_user_cluster.joblib",
    inplace: bool = False
) -> pd.DataFrame:
    """
    Takes a DataFrame with 1 or more rows containing the ORIGINAL input columns
    (same names as training), predicts cluster IDs using saved KMeans + scaling stats,
    and returns the original df with an added 'cluster_id' column only.
    """

    if not isinstance(user_df, pd.DataFrame):
        raise TypeError("user_df must be a pandas DataFrame.")

    if len(user_df) == 0:
        raise ValueError("user_df must contain at least 1 row.")

    # --- load artifacts ---
    artifacts = joblib.load(artifacts_path)
    kmeans = artifacts["kmeans"]
    numeric_cols = artifacts["numeric_cols"]
    mean_dict = artifacts["mean_dict"]
    std_dict = artifacts["std_dict"]
    features_no_diet = artifacts["features_no_diet"]

    # this is what we return
    df_out = user_df if inplace else user_df.copy()

    # --- create TEMP dataframe for model input ---
    df_tmp = df_out.copy()

    # --- scale numeric columns into temporary *_scaled columns ---
    for col in numeric_cols:
        if col not in df_tmp.columns:
            raise KeyError(f"Missing required numeric column: '{col}'")

        df_tmp[col] = pd.to_numeric(df_tmp[col], errors="raise")

        mu = mean_dict[col]
        std = std_dict[col]

        if std == 0 or pd.isna(std):
            df_tmp[col + "_scaled"] = 0.0
        else:
            df_tmp[col + "_scaled"] = (df_tmp[col] - mu) / std

    # --- validate final model features ---
    missing_features = [c for c in features_no_diet if c not in df_tmp.columns]
    if missing_features:
        raise KeyError(
            "Missing required feature columns for prediction: "
            + ", ".join(missing_features)
        )

    # --- build model input ---
    X = df_tmp[features_no_diet].values

    # --- predict ---
    cluster_ids = kmeans.predict(X)

    # --- attach ONLY the result ---
    df_out["cluster_id"] = cluster_ids

    return df_out


def create_user_features(df: pd.DataFrame, new_user: pd.DataFrame) -> pd.DataFrame:
    """
    Add to df the following
    
    user-level features:
    - BMR (Mifflin-St Jeor)
    - PAL = 1.2 + 0.175 * Workout_Frequency
    - TDEE = BMR * PAL

    Goal / calorie plan features:
    - CalorieChange = WeightChange * 7700
    - CaloriesToBurnTraining = CalorieChange * 0.5
    - CaloriesReducedFromFood = CalorieChange * 0.5
    - CaloriesPerDay = TDEE +/- (CaloriesReducedFromFood / GoalDays)  (minus for Loss, plus for Gain)
    - TotalWorkouts = Workout_Frequency * GoalDays / 7
    - CaloriesPerWorkout = CaloriesToBurnTraining / TotalWorkouts

    Expected columns from new_user:
    Age, Gender, Weight (kg), Height (m), BMI, Workout_Frequency (days/week),
    WeightChange (kg), GoalDays, Goal ({"Loss","Maintain","Gain"}), cluster_id
    """
    out = df.copy()
    # Basic fields
    weight = out["Weight (kg)"].astype(float)
    height_cm = (out["Height (m)"].astype(float) * 100.0)
    age = out["Age"].astype(float)

    # Gender: detect male vs female
    is_male = out["Gender"].astype(str).str.lower().str.startswith("m")

    # BMR 
    bmr_male = 10 * weight + 6.25 * height_cm - 5 * age + 5
    bmr_female = 10 * weight + 6.25 * height_cm - 5 * age - 161
    out["BMR"] = np.where(is_male, bmr_male, bmr_female)

    # PAL and TDEE
    wf = float(new_user["Workout_Frequency (days)"].iloc[0])
    out["PAL"] = 1.2 + 0.175 * wf
    out["TDEE"] = out["BMR"] * out["PAL"]

    goal = str(new_user["Goal"].iloc[0]).strip()
    goal_days = float(new_user["GoalDays"].iloc[0])
    weight_change_kg = float(abs(new_user["WeightChange (kg)"].iloc[0]))

    out["CalorieChange"] = weight_change_kg * 7700.0
    out["CaloriesToBurnTraining"] = out["CalorieChange"] * 0.5
    out["CaloriesReducedFromFood"] = out["CalorieChange"] * 0.5
    if goal == "Gain":
        out["CaloriesReducedFromFood"] = out["CalorieChange"] * 1.5

    daily_delta = (out["CaloriesReducedFromFood"] / goal_days) if goal_days > 0 else np.nan

    out["CaloriesPerDay"] = out["TDEE"]
    if goal == "Loss":
        out["CaloriesPerDay"] = out["TDEE"] - daily_delta
    elif goal == "Gain":
        out["CaloriesPerDay"] = out["TDEE"] + daily_delta
    else:
        out["CaloriesPerDay"] = out["TDEE"]
        out["CaloriesToBurnTraining"] = out["TDEE"] * out["GoalDays"] * 0.15

    out["TotalWorkouts"] = wf * (goal_days / 7.0)
    out["CaloriesPerWorkout"] = (
        out["CaloriesToBurnTraining"] / out["TotalWorkouts"]
        if out["TotalWorkouts"].iloc[0] > 0 else 0.0
    )
    if goal == "Gain":
        out["CaloriesPerDay"] = out["CaloriesPerDay"] + out["CaloriesToBurnTraining"] / out["GoalDays"]
    return out


def create_workout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Workout features:
    E - Energy Consumption
    I - Intensity
    S - Power component
    D - Duration (РЅРѕСЂРјРёСЂРѕРІР°РЅРЅР°СЏ)
    R - Risk (1 - penalties)
    """
    out = df.copy()
    #out["Cal_30_min"] = df["Burns Calories (per 30 min)"].copy()

    # --- E: Energy consumption ---
    # E_raw
    cals = out["Calories_Burned"]
    out["E_raw"] = (cals - cals.min()) / (cals.max() - cals.min())

    # E_eff
    burn30 = out["Burns Calories (per 30 min)"]
    out["E_eff"] = (burn30 - burn30.min()) / (burn30.max() - burn30.min())

    out["E"] = 0.5 * out["E_raw"] + 0.5 * out["E_eff"]

    # --- I: Intensity (HRR) ---
    out["pct_HRR"] = (out["Avg_BPM"] - out["Resting_BPM"]) / (out["Max_BPM"] - out["Resting_BPM"])
    hrr = out["pct_HRR"]
    out["I"] = (hrr - hrr.min()) / (hrr.max() - hrr.min())

    # --- S: Power component ---
    out["workload"] = out["Sets"] * out["Reps"]
    S = out["workload"] / out["Difficulty Level"]
    out["S"] = (S - S.min()) / (S.max() - S.min())

    # --- D: Duration (normalize training duration) ---
    out["Duration_min"] = out["Session_Duration (hours)"] * 60
    dur = out["Duration_min"]
    out["D"] = (dur - dur.min()) / (dur.max() - dur.min())

    # --- R: Risk  ---

    # 1) Age
    age = out["Age"]
    out["pen_age"] = (age - age.min()) / (age.max() - age.min())

    # 2) BMI
    bmi = out["BMI"]
    out["pen_bmi"] = (bmi - bmi.min()) / (bmi.max() - bmi.min())

    # 3) HRR (upper bound)
    hrr90 = hrr.quantile(0.9)
    hrr_max = hrr.max()
    out["pen_hrr"] = np.where(
        hrr > hrr90,
        (hrr - hrr90) / (hrr_max - hrr90),
        0.0
    )

    # 4) Skill penalty (difficulty > experience level)
    out["pen_skill"] = np.maximum(0, out["Difficulty Level"] - out["Experience_Level"])

    out["R"] = 1 - (
        0.4 * out["pen_age"]
      + 0.3 * out["pen_bmi"]
      + 0.2 * out["pen_hrr"]
      + 0.1 * out["pen_skill"]
    )
    out = out.drop(columns=["BMR", "PAL", "TDEE", "CalorieChange", "CaloriesToBurnTraining", "CaloriesReducedFromFood", 
                            "CaloriesPerDay", "TotalWorkouts", "CaloriesPerWorkout", "E_raw", "E_eff", "pct_HRR", "Session_Duration (hours)", 
                            "pen_age", "pen_bmi", "pen_hrr", "pen_skill"], errors='ignore') # drop helper variables created from other columns
    out = out.drop(columns=["Age", "BMI", "Experience_Level", "Difficulty Level", "Duration_min", 
                            "Max_BPM", "Avg_BPM", "Resting_BPM", "workload", "Burns Calories (per 30 min)"], errors='ignore') # drop these columns to avoid data leakage
    out = out.drop(columns=["cooking_method","meal_type", "Calories", "serving_size_g", "sugar_g", "sodium_g", "cholesterol_g", "Carbs", "Proteins", "Fats"], errors='ignore') # drop meal related columns, leave 'meal_name' only
    return out



def create_meal_features(df: pd.DataFrame, new_user: pd.DataFrame) -> pd.DataFrame:
    """
    Create meal features:

    C  -> Calorie Fit
    P  -> Protein per meal
    M  -> Macro Match
    ED -> Energy Density
    F  -> Food Safety
    """

    out = df.copy()
    out['Cal'] = out['Calories'] / out ["Daily meals frequency"]
    # Create 'meal_name' by combining cooking_method, diet_type, and meal_type
    out['meal_name'] = (
        out['cooking_method'].fillna('Unknown').astype(str) + " " +
        out['diet_type'].fillna('Balanced').astype(str) + " " +
        out['meal_type'].fillna('Meal').astype(str)
    ).str.title()

    # Drop the original columns after combining
    #out = out.drop(columns=['cooking_method', 'diet_type', 'meal_type'])

    # --- C: Calorie Fit ---
    calories_per_day = float(new_user["CaloriesPerDay"].iloc[0])
    meal_target = calories_per_day / out["Daily meals frequency"].astype(float)

    out["C"] = 1 - (
        (out["Calories"] / out["Daily meals frequency"] - meal_target).abs()
        / meal_target
    )

    # --- P: Proteins per meal ---
    P = out["Proteins"] / out["Daily meals frequency"]
    out["P"] = (P - P.min()) / (P.max() - P.min())

    # --- M: Macro Match ---
    out["cal_from_protein"] = out["Proteins"] * 4
    out["cal_from_carbs"] = out["Carbs"] * 4
    out["cal_from_fats"] = out["Fats"] * 9

    total_macro_cal = (
        out["cal_from_protein"]
        + out["cal_from_carbs"]
        + out["cal_from_fats"]
    ).replace(0, np.nan)

    out["pct_p"] = out["cal_from_protein"] / total_macro_cal
    out["pct_c"] = out["cal_from_carbs"] / total_macro_cal
    out["pct_f"] = out["cal_from_fats"] / total_macro_cal

    goal = new_user["Goal"].iloc[0]
    if goal == "Loss":
        target_p, target_c, target_f = 0.3, 0.35, 0.35
    elif goal == "Maintain":
        target_p, target_c, target_f = 0.2, 0.5, 0.3
    elif goal == "Gain":
        target_p, target_c, target_f = 0.25, 0.55, 0.2
    else:
        raise ValueError(f"Unknown goal: {goal}")

    out["M"] = 1 - (1 / 3) * (
        (out["pct_p"] - target_p).abs() / target_p
        + (out["pct_c"] - target_c).abs() / target_c
        + (out["pct_f"] - target_f).abs() / target_f
    )

    # --- ED: Energy Density ---
    ED = out["Calories"] / (out["serving_size_g"] * out["Daily meals frequency"])
    out["ED"] = (ED - ED.min()) / (ED.max() - ED.min())

    # --- F: Food Safety ---
    sugar_90 = out["sugar_g"].quantile(0.9)
    sodium_90 = out["sodium_g"].quantile(0.9)
    chol_90 = out["cholesterol_g"].quantile(0.9)

    out["F"] = 1 - (1 / 3) * (
        (out["sugar_g"] / sugar_90).clip(upper=1)
        + (out["sodium_g"] / sodium_90).clip(upper=1)
        + (out["cholesterol_g"] / chol_90).clip(upper=1)
    )

    out = out.drop(columns=["Calories_Burned", "Workout_Frequency (days)", "cholesterol_g", 
                            "sodium_g", "sugar_g", "Calories", "Daily meals frequency", "serving_size_g"]) # drop these features to avoid data leakage
    out = out.drop(columns=["CalorieChange", "CaloriesToBurnTraining", "CaloriesReducedFromFood", "CaloriesPerDay", "CaloriesPerWorkout", 
                            "TotalWorkouts", "pct_p", "pct_c", "pct_f", "BMR", "PAL", "TDEE", "cal_from_protein", "cal_from_carbs", "cal_from_fats"]) # drop these helper features
    out = out.drop(columns=['Sets', 'Reps', 'rating', 'Workout_Type', 'Name of Exercise', 'Benefit',
                            'Target Muscle Group', 'Equipment Needed', 'Body Part', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)',
                            'Type of Muscle', 'Workout', 'Experience_Level', 'Difficulty Level', 'Burns Calories (per 30 min)']) # workout
    return out



def build_and_train_workout_model(df: pd.DataFrame, new_user: pd.DataFrame) -> None:
    goal = new_user['Goal'].iloc[0]

    # -------------------------------
    # Target construction (allowed)
    # -------------------------------
    if goal == 'Loss':
        df['target'] = 0.45*df['E'] + 0.25*df['I'] + 0.10*df['D'] + 0.05*df['S'] + 0.15*df['R']
    elif goal == 'Maintain':
        df['target'] = 0.25*df['E'] + 0.20*df['I'] + 0.15*df['D'] + 0.20*df['S'] + 0.20*df['R']
    elif goal == 'Gain':
        df['target'] = 0.05*df['E'] + 0.15*df['I'] + 0.10*df['D'] + 0.50*df['S'] + 0.20*df['R']
    else:
        raise ValueError("Goal must be one of: 'Loss', 'Maintain', 'Gain'")

    # Drop target construction components
    df = df.drop(columns=['E', 'I', 'D', 'S', 'R'])

    # -------------------------------
    # Feature definitions
    # -------------------------------
    numerical_features = [
        'Weight (kg)', 'Height (m)',
        'Workout_Frequency (days)', 'Daily meals frequency', 'rating'
    ]

    categorical_features = [
        'Gender', 'Workout_Type', 'diet_type', 'Name of Exercise', 'Benefit',
        'Target Muscle Group', 'Equipment Needed', 'Body Part',
        'Type of Muscle', 'Workout', 'cluster_id'
    ]

    selected_features = numerical_features + categorical_features + ['target']

    # -------------------------------
    # create df_selected
    # calories_burned is NOT included here
    # -------------------------------
    df_selected = df[selected_features].copy()

    X = df_selected.drop(columns=['target'])
    y = df_selected['target']

    # -------------------------------
    # Train / Val / Test split
    # -------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # -------------------------------
    # Preprocessing
    # -------------------------------
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ],
        remainder="drop"
    )

    X_train_enc = preprocess.fit_transform(X_train)
    X_val_enc   = preprocess.transform(X_val)
    X_test_enc  = preprocess.transform(X_test)

    # -------------------------------
    # Helper
    # -------------------------------
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # ============================================================
    # SECTION 3: TUNED MODEL (RandomizedSearchCV on TRAIN only)
    # ============================================================
    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 800, 1200],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5, 6, 8],
        "min_child_weight": [1, 2, 5, 10],
        "subsample": [0.6, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5, 1.0],
        "reg_alpha": [0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }

    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train_enc, y_train)

    best_params = search.best_params_
    best_cv_rmse = np.sqrt(-search.best_score_)

    print("[Tuned HP] Best CV params:", best_params)
    print(f"[Tuned HP] Best CV RMSE: {best_cv_rmse:.6f}")

    # Validation check
    xgb_tuned_train_only = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    xgb_tuned_train_only.fit(X_train_enc, y_train)

    val_pred = xgb_tuned_train_only.predict(X_val_enc)
    print(f"[Tuned HP] Validation RMSE: {rmse(y_val, val_pred):.6f}")

    # ============================================================
    # FINAL MODEL: Train on TRAIN + VAL
    # ============================================================
    if issparse(X_train_enc) or issparse(X_val_enc):
        X_trainval_enc = vstack([X_train_enc, X_val_enc])
    else:
        X_trainval_enc = np.vstack([X_train_enc, X_val_enc])

    y_trainval = np.concatenate([y_train.to_numpy(), y_val.to_numpy()])

    final_model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_trainval_enc, y_trainval)

    test_pred = final_model.predict(X_test_enc)
    print(f"[FINAL] Test RMSE: {rmse(y_test, test_pred):.6f}")

    joblib.dump(preprocess, "encoder.pkl")
    joblib.dump(final_model, "models/workout_model.pkl")
    print("Workout model is trained!")



def predict_workout_score(df: pd.DataFrame, new_user: pd.DataFrame) -> pd.DataFrame:
    new_user = new_user.drop(
                    columns=['Age', 'Experience_Level', 'Goal', 'WeightChange (kg)', 'GoalDays', 'BMR', 'PAL', 'TDEE', 'CalorieChange', 'CaloriesToBurnTraining',
                        'CaloriesReducedFromFood', 'CaloriesPerDay',  'TotalWorkouts', 'CaloriesPerWorkout']
                )

    # Repeat user row to match number of exercises
    exercise_df_full = df[
                    ['Sets', 'Reps', 'rating', 'Workout_Type', 'Name of Exercise', 'Benefit',
                        'Target Muscle Group', 'Equipment Needed', 'Body Part',
                        'Type of Muscle', 'Workout', 'Calories_Burned']
    ].reset_index(drop=True)

    exercise_df_cleaned = exercise_df_full.drop(columns=['Sets', 'Reps', 'Calories_Burned'])

    # Repeat user row to match number of exercises
    workout_predict_cleaned = pd.concat(
        [pd.concat([new_user] * len(exercise_df_cleaned), ignore_index=True), exercise_df_cleaned],
        axis=1
    )

    workout_predict_full = pd.concat(
        [pd.concat([new_user] * len(exercise_df_full), ignore_index=True), exercise_df_full],
        axis=1
    )

    preprocess = joblib.load("encoder.pkl")
    final_model = joblib.load("models/workout_model.pkl")

    workout_predict_enc = preprocess.transform(workout_predict_cleaned)
    predictions = final_model.predict(workout_predict_enc)
    workout_predict_full["workout_score"] = predictions

    return workout_predict_full


def run_cosine_similarity_workout(df: pd.DataFrame, new_user: pd.DataFrame) -> pd.DataFrame:
    user_full = new_user.copy()
    workout_predict = predict_workout_score(df, new_user)
    new_user = new_user.drop(
                    columns=['Age', 'Goal', 'WeightChange (kg)', 'GoalDays', 'BMR', 'PAL', 'TDEE', 'CalorieChange', 'CaloriesToBurnTraining',
                        'CaloriesReducedFromFood', 'CaloriesPerDay',  'TotalWorkouts', 'CaloriesPerWorkout', "BMI"]
                )
    # ----------------------------
    # 1) Candidate generation: Top N
    # ----------------------------
    TOP_N = 100
    candidates = (workout_predict.sort_values("workout_score", ascending=False).head(TOP_N).reset_index(drop=True).copy())
    # ----------------------------
    # 2) Build text for vectorization (classic content-based)
    # ----------------------------
    TEXT_COLS = ["Benefit", "Target Muscle Group", "Equipment Needed", "Workout", "Body Part", "Type of Muscle", "Workout_Type", "Name of Exercise"]

    def make_text_profile(df: pd.DataFrame, cols=TEXT_COLS) -> pd.Series:
        tmp = df[cols].copy()
        for c in cols:
            tmp[c] = (tmp[c].astype(str).fillna("")
                    .str.lower()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip())
        return tmp.apply(lambda r: " ".join([v for v in r.values if v and v != "nan"]), axis=1)

    exercise_text = make_text_profile(candidates)

    # TF-IDF vectors for exercises
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    X_ex = vectorizer.fit_transform(exercise_text)
    # ----------------------------
    # 3) Create "day prototypes" as text -> vector
    # ----------------------------
    day_prototypes = {
        "Legs": "legs lower body quads quadriceps hamstrings glutes calves posterior",
        "Push": "chest shoulders triceps upper chest pressing push",
        "Pull": "back lats upper back biceps rows pull",
        "Core": "abs core obliques plank dead bug flutter kicks",
    }

    proto_text = list(day_prototypes.values())
    X_proto = vectorizer.transform(proto_text)  # same vector space

    # similarity of each exercise to each day prototype
    S_day = cosine_similarity(X_ex, X_proto)   # shape: (n_exercises, 4)

    day_names = list(day_prototypes.keys())
    candidates["day_label"] = [day_names[i] for i in np.argmax(S_day, axis=1)]
    candidates["day_sim"] = np.max(S_day, axis=1)  # confidence (how strongly it matches)

    # ----------------------------
    # 4) MMR reranking: relevant + diverse inside each day
    # ----------------------------
    def mmr_select_calories(df_day: pd.DataFrame, X_day_vecs, calories_target, k=4, alpha=0.75):
        if len(df_day) == 0:
            return df_day

        # normalize
        rel = df_day["workout_score"].to_numpy()
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)

        selected_idx = []
        remaining = list(range(len(df_day)))
        sims = cosine_similarity(X_day_vecs, X_day_vecs)

        current_calories = 0.0

        while remaining and len(selected_idx) < k and current_calories < calories_target:
            if not selected_idx:
                best = remaining[int(np.argmax(rel[remaining]))]
            else:
                mmr_scores = []
                for i in remaining:
                    max_sim_to_selected = np.max(sims[i, selected_idx]) if selected_idx else 0
                    mmr = alpha * rel[i] - (1-alpha) * max_sim_to_selected
                    mmr_scores.append(mmr)
                best = remaining[int(np.argmax(mmr_scores))]

            selected_idx.append(best)
            current_calories += df_day.iloc[best]["Calories_Burned"]  # summing calories burned
            remaining.remove(best)

            if current_calories >= calories_target:
                break  # if reached target workout calories (per day)

        return df_day.iloc[selected_idx].copy()

    # ----------------------------
    # 5) Plan Generation TotalWorkouts
    # ----------------------------
    plan = []
    for day in range(int(user_full.TotalWorkouts.iloc[0])):
        day_label = day_names[day % len(day_names)]
        df_day = candidates[candidates["day_label"] == day_label].copy()
        idx = df_day.index.to_numpy()
        X_day = X_ex[idx]

        day_plan = mmr_select_calories(df_day, X_day, calories_target=user_full.CaloriesPerWorkout.iloc[0], k=4, alpha=0.75)
        plan.append((day_label, day_plan))
    # ----------------------------
    # 6) Output
    # ----------------------------
    cols_show = ['Workout_Type', 'Workout', 'Name of Exercise', 'Body Part',
                 'Equipment Needed', 'Sets', 'Reps', 'Benefit', 'Calories_Burned']
    all_days = []
    interval = max(1, user_full.GoalDays.iloc[0] // user_full.TotalWorkouts.iloc[0])
    workout_days = list(range(1, int(user_full.GoalDays.iloc[0])+1, int(interval)))
    for i, (day_label, day_plan) in enumerate(plan, start=1):
        temp = day_plan[cols_show].copy()
        temp["Day"] = workout_days[i]
        all_days.append(temp)

    plan_df = pd.concat(all_days, ignore_index=True)
    plan_df.to_csv("workout_plan.csv", index=False)
    print("Plan saved -> workout_plan.csv")
    return plan_df


def save_to_pdf(meal_df, workout_df, user_df, filename="plan.pdf"):
    meal_df['Calories_Final'] = meal_df['Calories_Final'].round(2)
    workout_df['Calories_Burned'] = workout_df['Calories_Burned'].round(2)
    doc = SimpleDocTemplate(
        filename,
        pagesize=landscape(A4),
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20
    )

    styles = getSampleStyleSheet()
    elements = []
    # USER
    user_info = f"""
    <b>User Info:</b><br/>
    Age: {user_df['Age'].iloc[0]}<br/>
    Gender: {user_df['Gender'].iloc[0]}<br/>
    Weight (kg): {user_df['Weight (kg)'].iloc[0]}<br/>
    Height (m): {user_df['Height (m)'].iloc[0]}<br/>
    BMI: {round(user_df['BMI'].iloc[0], 2)}<br/>
    Goal: {user_df['Goal'].iloc[0]}<br/>
    Target weight change (kg): {user_df['WeightChange (kg)'].iloc[0]}<br/>
    """

    cal_meal = round(meal_df['Calories_Final'].sum(), 2)
    cal_burned = round(workout_df['Calories_Burned'].sum(), 2)
    total_energy_spent = (user_df["TDEE"].iloc[0] * user_df["GoalDays"].iloc[0] + cal_burned)
    actual_change = cal_meal - total_energy_spent
    weight_change = round(actual_change / 7700, 2)
    target_food = round(user_df.CaloriesPerDay.iloc[0] * user_df.GoalDays.iloc[0], 2)

    summary_text = f"""
    <b>Summary:</b><br/>
    Target meal calories: {target_food} | Actual: {cal_meal}<br/>
    Target workout burned calories: {round(user_df.CaloriesToBurnTraining.iloc[0], 2)} | Actual: {cal_burned}<br/>
    Target weight change (kg): {user_df['WeightChange (kg)'].iloc[0]} | Actual: {weight_change}<br/>
    """

    elements.append(Paragraph(user_info, styles["Normal"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))
    elements.append(Paragraph(summary_text, styles["Normal"]))
    elements.append(Paragraph("<br/><br/>", styles["Normal"]))


    all_days = sorted(set(meal_df["Day"]).union(set(workout_df["Day"])))

    for day in all_days:
        # DAY
        elements.append(Paragraph(f"<b>Day {day}</b>", styles["Heading2"]))

        day_meals = meal_df[meal_df["Day"] == day]
        day_workouts = workout_df[workout_df["Day"] == day]

        # FOOD
        if not day_meals.empty:
            meal_table_data = [day_meals.columns.tolist()] + day_meals.values.tolist()

            # --- Рассчет ширин колонок с масштабированием под страницу ---
            col_widths = []
            for col in day_meals.columns:
                max_len = max(day_meals[col].astype(str).apply(len).max(), len(col))
                col_widths.append(max_len * 0.4 * cm)

            total_width = sum(col_widths)
            if total_width > doc.width:
                scale = doc.width / total_width
                col_widths = [w * scale for w in col_widths]

            meal_table = Table(meal_table_data, repeatRows=1, colWidths=col_widths)
            meal_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]))
            elements.append(meal_table)
        else:
            elements.append(Paragraph("No meals", styles["Normal"]))
        elements.append(Paragraph("<br/>", styles["Normal"]))

        # WORKOUTS
        if not day_workouts.empty:
            elements.append(Paragraph("<br/><br/>", styles["Normal"]))
            workout_table_data = [day_workouts.columns.tolist()] + day_workouts.values.tolist()

            # --- Рассчет ширин колонок с масштабированием под страницу ---
            col_widths = []
            for col in day_workouts.columns:
                max_len = max(day_workouts[col].astype(str).apply(len).max(), len(col))
                col_widths.append(max_len * 0.4 * cm)

            total_width = sum(col_widths)
            if total_width > doc.width:
                scale = doc.width / total_width
                col_widths = [w * scale for w in col_widths]

            workout_table = Table(workout_table_data, repeatRows=1, colWidths=col_widths)
            workout_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]))
            elements.append(workout_table)
        else:
            elements.append(Paragraph("No workouts", styles["Normal"]))
        elements.append(Paragraph("<br/><br/>", styles["Normal"]))

    doc.build(elements)


ALLOWED_PORTIONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

def adjust_portions_per_meal(df: pd.DataFrame, target_calories: float) -> pd.DataFrame:
    df = df.copy()
    df["Portion"] = 1.0
    df["Calories_Final"] = df["Calories"]
    df = df.sort_values("Calories", ascending=False).reset_index(drop=True)

    current_total = df["Calories_Final"].sum()

    while current_total < target_calories:
        changed = False
        for i in range(len(df)):
            current_portion = df.loc[i, "Portion"]
            if current_portion < 3.0:
                new_portion = current_portion + 0.5

                if new_portion in ALLOWED_PORTIONS:
                    delta = df.loc[i, "Calories"] * 0.5
                    df.loc[i, "Portion"] = new_portion
                    df.loc[i, "Calories_Final"] += delta

                    current_total += delta
                    changed = True

                    if current_total >= target_calories:
                        break
        if not changed:
            break

    return df


def build_and_train_meal_model(df: pd.DataFrame, user: pd.DataFrame) -> None:
    goal = user['Goal'].iloc[0]

    if goal == 'Loss':
        df['target'] = 0.35*df['C'] + 0.25*df['P'] + 0.15*df['M'] + 0.15*df['ED'] + 0.1*df['F']
    elif goal == 'Maintain':
        df['target'] = 0.3*df['C'] + 0.25*df['M'] + 0.2*df['P'] + 0.1*df['ED'] + 0.15*df['F']
    elif goal == 'Gain':
        df['target'] = 0.4*df['C'] + 0.3*df['P'] + 0.15*df['M'] + 0.1*df['ED'] + 0.05*df['F']
    else:
        raise ValueError("Goal must be one of: 'Loss', 'Maintain', 'Gain'")
    df = df.drop(columns=['C', 'P', 'M', 'ED', 'F'])
    df_full = df.copy()
    # 3) Split X/y
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    numerical_features = ['Age', 'Weight (kg)', 'Height (m)', 'Water_Intake (liters)', 'BMI']

    categorical_features = ['Gender', 'meal_type', 'diet_type', 'cooking_method', 'cluster_id', 'meal_name']
    # 4) One-hot encoding (fit on train only, transform val/test)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ],
        remainder="drop"
    )

    X_train_enc = preprocess.fit_transform(X_train)
    X_val_enc   = preprocess.transform(X_val)
    X_test_enc  = preprocess.transform(X_test)
    # --------------------------------------------------
    # Manual selection of n_estimators using VALIDATION
    # --------------------------------------------------

    n_estimators_list = [50, 100, 200, 300, 500]
    results = []

    for n in n_estimators_list:
        model = XGBRegressor(
            n_estimators=n,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )
        
        model.fit(X_train_enc, y_train)
        val_pred = model.predict(X_val_enc)
        
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        results.append((n, rmse))

    # --------------------------------------------------
    # Select best n_estimators
    # --------------------------------------------------

    best_n, best_rmse = min(results, key=lambda x: x[1])
    # --------------------------------------------------
    # Train FINAL model on training data
    # --------------------------------------------------

    final_model = XGBRegressor(
        n_estimators=best_n,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    final_model.fit(X_train_enc, y_train);  
    test_pred = final_model.predict(X_test_enc)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print(f"Test RMSE: {test_rmse:.4f}")
    # Base model (we keep objective fixed for regression)
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    # Parameter search space (covers the big knobs)
    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 800, 1200],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5, 6, 8],
        "min_child_weight": [1, 2, 5, 10],
        "subsample": [0.6, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5, 1.0],
        "reg_alpha": [0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }

    # Use neg MSE for max compatibility with older sklearn,
    # then we take sqrt later for RMSE.
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=40,                 # increase to 80+ if you can afford time
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train_enc, y_train)

    print("Best CV params:", search.best_params_)
    print("Best CV RMSE:", np.sqrt(-search.best_score_))
    best_model = search.best_estimator_

    val_pred = best_model.predict(X_val_enc)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"Validation RMSE after tuning: {val_rmse:.6f}")
    test_pred = best_model.predict(X_test_enc)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print(f"Test RMSE: {test_rmse:.6f}")
    joblib.dump(preprocess, "meal_encoder.pkl")
    joblib.dump(final_model, "models/meal_model.pkl")
    print("Saved the meal model!")

def predict_meal_score(df: pd.DataFrame, user: pd.DataFrame) -> pd.DataFrame:
    user_full = user.copy()
    user = user.drop(
                        columns=['Goal', 'WeightChange (kg)', 'GoalDays', 'BMR', 'PAL', 'TDEE', 'CalorieChange', 'CaloriesToBurnTraining', 
                            'CaloriesReducedFromFood', 'CaloriesPerDay',  'TotalWorkouts', 'CaloriesPerWorkout', 'Experience_Level', 
                            'Workout_Frequency (days)', 'Daily meals frequency']
                )
    # Meal-related features
    meal_df = df[
                        ['Water_Intake (liters)', 'meal_type', 'cooking_method', 'meal_name' ]
    ].reset_index(drop=True)

    # Repeat user row to match number of meals 
    meal_predict = pd.concat(
        [pd.concat([user] * len(meal_df), ignore_index=True), meal_df],
        axis=1
    )
    preprocess = joblib.load("meal_encoder.pkl")
    final_model = joblib.load("models/meal_model.pkl")

    meal_predict_enc = preprocess.transform(meal_predict)
    predictions = final_model.predict(meal_predict_enc)
    meal_predict = meal_predict.copy()
    meal_predict["meal_score"] = predictions

    meal_predict["Calories"] = df["Cal"].values
    meal_predict["Proteins"] = df['Proteins'].values
    meal_predict["Carbs"] = df['Carbs'].values
    meal_predict["Fats"] = df['Fats'].values
    return meal_predict



def run_cosine_similarity_meal(df: pd.DataFrame, new_user: pd.DataFrame) -> pd.DataFrame:
    meal_predict = predict_meal_score(df, new_user)
    # ----------------------------
    # 1) Candidate generation: Top N meals
    # ----------------------------
    TOP_N = 100
    candidates = (
        meal_predict
        [meal_predict["diet_type"] == new_user['diet_type'].iloc[0]]
        .sort_values("meal_score", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
        .copy()
    )

    # ----------------------------
    # 2) Build text for vectorization (optional, e.g., for meal type / diet_type)
    # ----------------------------
    TEXT_COLS = ["meal_name", "diet_type", "cooking_method"]  # можно расширять

    def make_text_profile(df: pd.DataFrame, cols=TEXT_COLS) -> pd.Series:
        tmp = df[cols].copy()
        for c in cols:
            tmp[c] = (tmp[c].astype(str).fillna("")
                    .str.lower()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip())
        return tmp.apply(lambda r: " ".join([v for v in r.values if v and v != "nan"]), axis=1)

    meal_text = make_text_profile(candidates)

    # TF-IDF vectors for meals
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    X_meal = vectorizer.fit_transform(meal_text)

    # ----------------------------
    # 3) Create "day prototypes" (breakfast/lunch/dinner/snack or day parts)
    # ----------------------------
    day_prototypes = {
        "Breakfast": "breakfast",
        "Lunch": "lunch",
        "Dinner": "dinner",
        "Snack": "snack",
    }

    proto_text = list(day_prototypes.values())
    X_proto = vectorizer.transform(proto_text)

    # similarity of each meal to each day prototype
    S_day = cosine_similarity(X_meal, X_proto)

    day_names = list(day_prototypes.keys())
    candidates["day_label"] = [day_names[i] for i in np.argmax(S_day, axis=1)]
    candidates["day_sim"] = np.max(S_day, axis=1)

    # ----------------------------
    # 4) MMR reranking: relevant + diverse + calories target per meal
    # ----------------------------
    def mmr_select_calories(df_day: pd.DataFrame, X_day_vecs, calories_target, k=4, alpha=0.75):
        if len(df_day) == 0:
            return df_day

        rel = df_day["meal_score"].to_numpy()
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)

        selected_idx = []
        remaining = list(range(len(df_day)))
        sims = cosine_similarity(X_day_vecs, X_day_vecs)
        current_calories = 0.0

        while remaining and len(selected_idx) < k and current_calories < calories_target:
            if not selected_idx:
                best = remaining[int(np.argmax(rel[remaining]))]
            else:
                mmr_scores = []
                for i in remaining:
                    max_sim_to_selected = np.max(sims[i, selected_idx]) if selected_idx else 0
                    mmr = alpha * rel[i] - (1 - alpha) * max_sim_to_selected
                    mmr_scores.append(mmr)
                best = remaining[int(np.argmax(mmr_scores))]

            selected_idx.append(best)
            current_calories += df_day.iloc[best]["Calories"]
            remaining.remove(best)
            if current_calories >= calories_target:
                break

        return df_day.iloc[selected_idx].copy()

    # ----------------------------
    # 5) Generate meal plan for all days
    # ----------------------------
    target_food = float(new_user["CaloriesPerDay"].iloc[0] * new_user["GoalDays"].iloc[0])
    goal_days = int(new_user['GoalDays'].iloc[0])
    target_per_meal = float(new_user['CaloriesPerDay'].iloc[0] / new_user['Daily meals frequency'].iloc[0])
    meals_per_day = int(new_user["Daily meals frequency"].iloc[0])

    plan = []
    remaining_calories = target_food

    candidates_sorted = candidates.sort_values("meal_score", ascending=False).copy()
    X_candidates = X_meal[candidates_sorted.index]

    selected_idx = []
    current_calories = 0.0
    alpha = 0.75  
    sims = cosine_similarity(X_candidates, X_candidates)
    rel = candidates_sorted["meal_score"].to_numpy()
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)

    while current_calories < remaining_calories and len(selected_idx) < len(candidates_sorted):
        if not selected_idx:
            best = int(np.argmax(rel))
        else:
            mmr_scores = []
            for i in range(len(candidates_sorted)):
                if i in selected_idx:
                    mmr_scores.append(-np.inf)
                    continue
                max_sim_to_selected = np.max(sims[i, selected_idx]) if selected_idx else 0
                mmr = alpha * rel[i] - (1 - alpha) * max_sim_to_selected
                mmr_scores.append(mmr)
            best = int(np.argmax(mmr_scores))
        
        selected_idx.append(best)
        current_calories += candidates_sorted.iloc[best]["Calories"]

    final_plan = candidates_sorted.iloc[selected_idx].copy()
    final_plan["Portion"] = 0.5
    final_plan["Calories"] = final_plan["Calories"]/2

    day_names = ["Breakfast", "Lunch", "Dinner", "Snack"]

    rows = []
    idx = 0

    for day in range(1, goal_days + 1):
        day_meals = final_plan.iloc[idx:idx + meals_per_day].copy()

        if len(day_meals) < meals_per_day:
            extra = final_plan.iloc[:meals_per_day - len(day_meals)]
            day_meals = pd.concat([day_meals, extra])

        day_meals["Day"] = day
        day_meals["day_label"] = day_names[:meals_per_day]

        rows.append(day_meals)
        idx += meals_per_day

    meal_plan_df = pd.concat(rows, ignore_index=True)
    meal_plan_df = adjust_portions_per_meal(
        meal_plan_df,
        target_calories=target_food
    )

    meal_plan_df = meal_plan_df.sort_values("Day").reset_index(drop=True)
    cols_show = ["meal_name", "diet_type", "Calories_Final", "day_label", "Day", "Portion", "Proteins", "Carbs", "Fats"]
    meal_plan_df = meal_plan_df[cols_show]
    meal_plan_df.to_csv("meal_plan.csv", index=False)
    print("Plan saved -> meal_plan.csv")
    return meal_plan_df


def generate_workout_plan(new_user: pd.DataFrame) -> pd.DataFrame:
    data = "data/dataset_with_user_features.csv" 
    df = pd.read_csv(data)
    df = create_workout_features(df)
    build_and_train_workout_model(df, new_user)
    df = run_cosine_similarity_workout(df, new_user)
    return df


def generate_meal_plan(new_user: pd.DataFrame) -> pd.DataFrame:
    data = "data/dataset_with_user_features.csv" 
    df = pd.read_csv(data)
    df = create_meal_features(df, new_user)
    build_and_train_meal_model(df, new_user)
    df = run_cosine_similarity_meal(df, new_user)
    return df

def check():
    user = pd.read_csv('data/new_user.csv')
    meals = pd.read_csv('meal_plan.csv')
    workouts = pd.read_csv('workout_plan.csv')
    cal_meal = round(meals['Calories_Final'].sum(), 2)
    cal_burned = round(workouts['Calories_Burned'].sum(), 2)
    total_energy_spent = (user["TDEE"].iloc[0] * user["GoalDays"].iloc[0] + cal_burned)
    actual_change = cal_meal - total_energy_spent
    target_food = round(user.CaloriesPerDay.iloc[0] * user.GoalDays.iloc[0], 2) 
    print('Target meal calories:', target_food, 'Actual:', cal_meal)
    print('Target workout burned calories:', round(user.CaloriesToBurnTraining.iloc[0], 2), "Actual:", cal_burned)
    weight_change = round(actual_change/7700, 2)
    print('Target weight change:', user['WeightChange (kg)'].iloc[0], 'Actual:', weight_change)
