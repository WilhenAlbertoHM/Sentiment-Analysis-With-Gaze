import subprocess
import os

"""
analyze_files: Analyzes all files found in rootdir that end with suffix using OpenFace located at openface. 
               The output is stored in outputdir.
Params: openface - The path to OpenFace.
        rootdir - The path to the root directory.
        outputdir - The path to the output directory.
        suffix - The suffix of the files to analyze.
Returns: None.
"""
def analyze_files(openface, rootdir, outputdir, suffix = ''):
    list_of_files = []
    subdirs = []
    
    # Walk through files and subdirectories in root.
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # If suffix is specified: search for files with suffix.
            if suffix != '':
                if file.endswith(suffix):
                    list_of_files.append(file)
                    subdirs.append(subdir)
            # Otherwise, use every file found.
            else:
                list_of_files.append(file)
                subdirs.append(subdir)
                
    print("Found "  + str(len(list_of_files)) + " files to analyze.")
    print("Starting analyzing process...")

    # If there are files to analyze, then run FaceLandmarkVidMulti on every video.
    if(len(list_of_files) > 0):
        i = 0
        for file in list_of_files:
            print("\nStarted analyzing file " + str(i+1) + "\n...")
            video_file_path = os.path.join(subdirs[i], file)
            
            # Ensure the video file exists.
            if os.path.exists(video_file_path):
                command = [openface, '-f', video_file_path, '-out_dir', 'processed']
                subprocess.run(command, shell=True)
                print("Finished analyzing file " + str(i+1))
            else:
                print(f"Video file not found: {video_file_path}")

            # Delete files that are not .csv files.
            for root, dirs, files in os.walk(outputdir):
                for f in files:
                    if ".csv" not in f:
                        os.remove(os.path.join(root, f))
                        print(f"Deleted: {f}")
                        
            print("Finished analyzing file " + str(i + 1))
            i += 1
        print("\nAll files are analyzed! The output is found in " + outputdir)  
    else:
        print("No files to analyze.")

"""
main: The main function.
"""
def main():
    open_dir = "OpenFace_2.2.0_win_x64/FaceLandmarkVidMulti.exe"
    root_dir = "data"
    output_dir = "processed"
    analyze_files(open_dir, root_dir, output_dir)

if __name__ == "__main__":
    main()