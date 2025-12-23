from __future__ import annotations
import pandas as pd
import numpy as np
from joblib import load


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


def save_user_to_csv(user_data: dict, output_path: str = "data/new_user.csv") -> None:
    df_new = pd.DataFrame([user_data])
    df_new.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

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
    artifacts = load(artifacts_path)
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
    WeightChange (kg), GoalDays, Goal ({"Loss","Maintain","Gain"})
    """
    out = df.copy()
    out = df.drop(columns=["Water_Intake (liters)"])
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

    daily_delta = (out["CaloriesReducedFromFood"] / goal_days) if goal_days > 0 else np.nan

    out["CaloriesPerDay"] = out["TDEE"]
    if goal == "Loss":
        out["CaloriesPerDay"] = out["TDEE"] - daily_delta
    elif goal == "Gain":
        out["CaloriesPerDay"] = out["TDEE"] + daily_delta

    out["TotalWorkouts"] = wf * (goal_days / 7.0)
    out["CaloriesPerWorkout"] = (
        out["CaloriesToBurnTraining"] / out["TotalWorkouts"]
        if out["TotalWorkouts"].iloc[0] > 0 else 0.0
    )


    return out


def create_workout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Workout features:
    E - Energy Consumption
    I - Intensity
    S - Power component
    D - Duration (нормированная)
    R - Risk (1 - penalties)
    """
    out = df.copy()

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
                            "pen_age", "pen_bmi", "pen_hrr", "pen_skill"]) # drop helper variables created from other columns
    out = out.drop(columns=["Age", "Experience_Level", "Difficulty Level", "Calories_Burned", "Burns Calories (per 30 min)", "Duration_min", 
                            "Max_BPM", "Avg_BPM", "Resting_BPM"]) # drop these columns to avoid data leakage
    out = out.drop(columns=["cooking_method","meal_type", "Calories", "serving_size_g", "sugar_g", "sodium_g", "cholesterol_g", "Carbs", "Proteins", "Fats"]) # drop meal related columns, leave 'meal_name' only
    return out



def create_meal_features(df: pd.DataFrame, new_user: pd.DataFrame) -> pd.DataFrame:
    """
    Create meal features:

    C  – Calorie Fit
    P  – Protein per meal
    M  – Macro Match
    ED – Energy Density
    F  – Food Safety
    """

    out = df.copy()
    # Create 'meal_name' by combining cooking_method, diet_type, and meal_type
    out['meal_name'] = (
        out['cooking_method'].fillna('Unknown').astype(str) + " " +
        out['diet_type'].fillna('Balanced').astype(str) + " " +
        out['meal_type'].fillna('Meal').astype(str)
    ).str.title()

    # Drop the original columns after combining
    out = out.drop(columns=['cooking_method', 'diet_type', 'meal_type'])

    # --- C: Calorie Fit ---
    meal_target = new_user["Meal_target"].iloc[0]
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

    return out





