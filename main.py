from functions import *

def main() -> None:
    user_dict = collect_new_user()              # dict
    df = pd.DataFrame([user_dict])              # 1-row DataFrame

    df = predict_cluster_id(df)                 # add cluster_id 
    df = create_user_features(df, df)             # add BMR/TDEE/etc
    
    df.to_csv('data/new_user.csv', index=False)
# generate_workout_plan

if __name__ == "__main__":
    main()
