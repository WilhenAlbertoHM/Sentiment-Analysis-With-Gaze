import pandas as pd
import os

"""
merge_csv: Merges multiple .csv files into one .csv file with 
           the names of the videos corresponding to the data.
Params: cols - A list of columns that we're using.
        folder_with_all_csvs - A folder with all the .csv files that we're merging.
        combined_csv_path - A path to the combined .csv file.
Returns: None.
"""
def merge_csv(cols, folder_with_all_csvs, combined_csv_path):
    if not os.path.exists(folder_with_all_csvs):
        print(f"Folder {folder_with_all_csvs} does not exist.")
        return

    # Iterate through all the .csv files in the directory, 
    # make dataframes out of them, and append them to the list.
    dataframes = []
    for file in os.listdir(folder_with_all_csvs):
        file_path = os.path.join(folder_with_all_csvs, file)
        df = pd.read_csv(file_path)
        new_csv_name = file.strip(".csv") + ".mp4"
        df["file_name"] = new_csv_name

        # Filter out the columns that we don't need, and reduce the memory usage.
        df = filter_df(df, cols)
        dataframes.append(df)

    # Concatenate all the dataframes into one dataframe and export to a .csv file.
    print("Concatenating all dataframes...")
    combined_data_df = pd.concat(dataframes, ignore_index=True)
    export_to_csv(combined_data_df, combined_csv_path)
    print(f"Combined data saved to {combined_csv_path}")

"""
filter_df: From a given dataframe, filter out the columns that we don't need, 
           and reduce the memory usage.
Params: df - A dataframe
       cols - A list of columns that we're using; remove the rest of columns.
Returns: A new dataframe with filtered columns.
"""
def filter_df(df, cols):
    # Use float16 to reduce memory usage.
    # Note: "file_name" is not of float type, so exclude it from turning it into float16.
    df = df[cols]
    df.iloc[:, :-1] = df.iloc[:, :-1].astype("float16")
    df.columns = df.columns.str.strip()
    return df

"""
add_personality_scores_col: Returns a new Dataframe with added personality scores 
                            to the eye gaze and head/neck rotation data.
Params: gaze_rotation_df - A dataframe with eye gaze and head/neck rotation data
        personalities_df - A dataframe with personality scores for each video
Returns: A new Dataframe with new columns for scores for each personality trait based on the video.
"""
def add_personality_scores_col(gaze_rotation_df, personalities_df):
    # Merge the two dataframes on the "vid_name" column;
    # this adds the personality scores per video, accordingly.
    merged_df = pd.merge(gaze_rotation_df, personalities_df, 
                         left_on="file_name", right_on="vid_name", how="left")
    merged_df.drop(columns=["vid_name"], inplace=True)
    return merged_df

"""
export_to_csv: Exports df to a .csv file.
Params: df - Dataframe that we're exporting.
        file_name - Full path to the .csv file.
Returns: None.
"""
def export_to_csv(df, file_name):
    df.to_csv(file_name, index=False)

"""
main: The main function.
"""
def main():
    # Get all the columns that we're using.
    cols = ["frame", "face_id", "timestamp", "confidence", "success", 
            "gaze_0_x", "gaze_0_y", "gaze_0_z",
            "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", 
            "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz", "file_name"]

    # Get the merged .csv file with eye gaze and head/neck rotation.
    folder_with_all_csvs = "processed"
    combined_csv_path = "csv_files/eye_gaze_rotation_with_vidnames_data.csv"
    annotations_path = "csv_files/annotations.csv"
    merge_csv(cols, folder_with_all_csvs, combined_csv_path)
    gaze_rotation_df = pd.read_csv(combined_csv_path)
    personalities_df = pd.read_csv(annotations_path)

    # Add personality scores to the eye gaze and head/neck rotation data,
    # and export to .csv.
    df = add_personality_scores_col(gaze_rotation_df, personalities_df)
    export_to_csv(df, "csv_files/gaze_rotation_and_scores_data.csv")

if __name__ == "__main__":
    main()