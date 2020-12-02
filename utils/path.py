import os

def is_image(path):
    path = path.lower()
    for ext in ['jpg', 'jpeg', 'png']:
        if path.endswith(ext):
            return True
    return False

def get_subfolders_with_files(path, is_file_func, yield_by_one=False):
    for dp, dn, fn in os.walk(path):
        file_paths = [os.path.join(dp, f) for f in fn if is_file_func(f)]
        if len(file_paths):
            if yield_by_one:
                for file_path in file_paths:
                    yield file_path
            else:
                yield file_paths