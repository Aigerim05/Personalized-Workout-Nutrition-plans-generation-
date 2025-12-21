from functions import *


def main() -> None:
    user_dict = collect_new_user()              # dict
    df = pd.DataFrame([user_dict])              # 1-row DataFrame

    df = create_user_features(df)               # add BMR/TDEE/etc
    df = predict_cluster_id(df)                 # add cluster_id (+ scaled cols)


if __name__ == "__main__":
    main()
