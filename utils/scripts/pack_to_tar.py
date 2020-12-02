import subprocess
import os



if __name__ == "__main__":

    out_zip_path = "pans_with_signs_2.tar.gz"

    with open("lat_lon_with_signs.txt") as in_file:
        old_folder_paths = set(map(lambda s: s.strip(), in_file.readlines()))
    
    with open("lat_lon_with_signs_2.txt") as in_file:
        new_folder_paths = set(map(lambda s: s.strip(), in_file.readlines()))
    
    # Remove old from list
    new_folder_paths = new_folder_paths - old_folder_paths

    print('Number of folders to zip:', len(new_folder_paths))
    
    paths_to_zip = []
    for folder_path in new_folder_paths:
        # Remove whitespace
        folder_path = folder_path.strip()
        # Split into parts
        out_path, lat_lon_path = os.path.split(folder_path)

        # Skip others
        if out_path != "out_vlad":
            continue
        
        paths_to_zip.append(
            os.path.join("out", lat_lon_path)
        )
    

    subprocess_command = [
        "tar",
        "-cvzf",
        out_zip_path,
        *paths_to_zip
    ]

    subprocess.Popen(subprocess_command)
