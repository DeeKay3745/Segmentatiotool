import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, help="Folder path to read files from")
parser.add_argument("--output", default="file_names.txt", help="Output txt file name")
parser.add_argument("--full_path", action="store_true", help="Save full paths instead of only file names")

args = parser.parse_args()

folder_path = args.folder
output_txt = args.output

if not os.path.isdir(folder_path):
    print(f"[ERROR] Folder does not exist: {folder_path}")
    exit(1)

with open(output_txt, "w", encoding="utf-8") as f:
    for name in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, name)

        if os.path.isfile(full_path):
            if args.full_path:
                f.write(full_path + "\n")
            else:
                f.write(name + "\n")

print(f"[DONE] File names saved in: {output_txt}")