#!/bin/bash

# -------- CONFIG --------
DEST_DIR="/lib/tensor"
# ------------------------

# 1. Check if argument provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_c_file>"
    exit 1
fi

CFILE="$1"

# 2. Check if path exists
if [ ! -e "$CFILE" ]; then
    echo "Error: Path does not exist."
    exit 1
fi

# 3. Check if it's a regular file
if [ ! -f "$CFILE" ]; then
    echo "Error: Not a valid file."
    exit 1
fi

# 4. Check if it's a .c file
if [[ "$CFILE" != *.c ]]; then
    echo "Error: Not a valid C source file (.c required)."
    exit 1
fi

# Extract filename without extension
FILENAME=$(basename "$CFILE" .c)
SONAME="lib${FILENAME}.so"

echo "Compiling $CFILE → $SONAME ..."

# 5. Compile shared object
gcc -shared -fPIC "$CFILE" -o "$SONAME"

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Compilation successful."

# 6. Create destination directory if not exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating $DEST_DIR ..."
    sudo mkdir -p "$DEST_DIR"
fi

# 7. Move the .so file
echo "Moving $SONAME to $DEST_DIR ..."
sudo mv "$SONAME" "$DEST_DIR/"

if [ $? -ne 0 ]; then
    echo "Failed to move .so file."
    exit 1
fi

echo "Done. Shared library installed at $DEST_DIR/$SONAME"
