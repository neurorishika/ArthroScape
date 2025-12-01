# arthroscape/rdp_client.py
"""
RDP Client module for secure data handling.

This module provides functionality to zip, encrypt, decrypt, and unzip folders and files.
It supports both single-file and multi-file (split) archives, using Fernet symmetric encryption.
"""

from cryptography.fernet import Fernet
import zipfile
import os
import argparse
from split_file_reader import SplitFileReader
from split_file_reader.split_file_writer import SplitFileWriter


def unlock_and_unzip_file(
    data2unzip: str, key_dir: str = "key.key", multifile: bool = False
) -> None:
    """
    Decrypt and unzip a file or a set of split files.

    This function takes an encrypted zip file (extension .ezip), decrypts it using the provided key,
    and then extracts its contents. It handles both single files and multi-part split files.

    Args:
        data2unzip (str): Path to the encrypted file (e.g., 'data.ezip' or 'data.ezip.000').
        key_dir (str): Path to the file containing the encryption key. Defaults to 'key.key'.
        multifile (bool): Whether the input is a multi-part split archive. Defaults to False.

    Raises:
        AssertionError: If the file extension or format does not match RDP standards.
        FileNotFoundError: If the key file or input file is not found.
    """
    # check if key exists
    try:
        with open(key_dir, "rb") as key_file:
            key = key_file.read()
        fernet = Fernet(key)
    except Exception as e:
        print(
            f"Key error: {e}. Please generate a key using generate_key(), provide an existing key_dir, or locate the key."
        )
        return

    if not multifile:
        # make sure its a ezip file
        assert (
            data2unzip.split(".")[-1] == "ezip"
        ), "data2unzip is not a ezip file under RDP standards"
        # make sure its not multifile
        assert (
            len(data2unzip.split("/")[-1].split(".")) == 2
        ), "data2unzip is a multifile ezip file, please set multifile=True"

        # decrypt file
        with open(data2unzip, "rb") as file:
            encrypted_data = file.read()
        decrypted_data = fernet.decrypt(encrypted_data)
        with open(data2unzip.replace("ezip", "zip"), "wb") as file:
            file.write(decrypted_data)

        # unzip all files and folders
        with zipfile.ZipFile(data2unzip.replace("ezip", "zip"), "r") as zip_ref:
            zip_ref.extractall(data2unzip[:-5])

        # delete zip file
        os.remove(data2unzip.replace("ezip", "zip"))
    else:
        # make sure its not singlefile
        assert (
            len(data2unzip.split("/")[-1].split(".")) == 3
        ), "data2unzip is a singlefile ezip file, please set multifile=False"
        # make sure its the first ezip file
        assert (
            data2unzip.split("/")[-1].split(".")[1] == "ezip"
            and data2unzip.split("/")[-1].split(".")[2] == "000"
        ), "data2unzip is not a multisplit ezip file under RDP standards"

        # get each split file
        split_files = list(
            filter(lambda x: x.startswith(data2unzip[:-4]), os.listdir())
        )

        # decrypt each split file
        for split_file in split_files:
            with open(split_file, "rb") as file:
                encrypted = file.read()
            decrypted = fernet.decrypt(encrypted)
            with open(split_file.replace("ezip", "zip"), "wb") as decrypted_file:
                decrypted_file.write(decrypted)

        # unzip all files and folders
        with SplitFileReader(
            [files.replace("ezip", "zip") for files in split_files]
        ) as sub_zip:
            with zipfile.ZipFile(sub_zip, "r") as zip_ref:
                zip_ref.extractall(data2unzip[:-9])

        # delete zip files
        for split_file in split_files:
            os.remove(split_file.replace("ezip", "zip"))


def zip_and_lock_folder(
    data2zip: str,
    key_dir: str = "key.key",
    multifile: bool = False,
    split_size_bytes: int = 50_000_000,
) -> None:
    """
    Zip and encrypt a folder.

    This function compresses a folder into a zip archive (or multiple split archives),
    encrypts the archive(s) using the provided key, and saves them with an .ezip extension.

    Args:
        data2zip (str): Path to the folder to zip and encrypt.
        key_dir (str): Path to the file containing the encryption key. Defaults to 'key.key'.
        multifile (bool): Whether to split the archive into multiple files. Defaults to False.
        split_size_bytes (int): Maximum size of each split file in bytes. Defaults to 50,000,000 (50MB).

    Raises:
        AssertionError: If data2zip is not a directory.
    """
    # check if key exists
    try:
        with open(key_dir, "rb") as key_file:
            key = key_file.read()
            fernet = Fernet(key)
    except Exception as e:
        print(
            f"Key error: {e}. Please generate a key using generate_key(), provide an existing key_dir, or locate the key."
        )
        return

    # make sure data2zip is a folder
    assert os.path.isdir(data2zip), "data2zip is not a folder"

    if not multifile:
        # zip folder while preserving directory structure
        with zipfile.ZipFile(
            f"{data2zip}.zip", "w"
        ) as fullzip:  # create zipfile object
            rootlen = (
                len(data2zip) + 1
            )  # get number of characters to remove from each file path
            for folder, subfolders, files in os.walk(
                f"{data2zip}"
            ):  # walk through folders
                for file in files:
                    fn = os.path.join(folder, file)
                    fullzip.write(fn, fn[rootlen:], compress_type=zipfile.ZIP_DEFLATED)
                for subfolder in subfolders:
                    fn = os.path.join(folder, subfolder)
        # encrypt zip file
        with open(f"{data2zip}.zip", "rb") as file:
            original = file.read()
        encrypted = fernet.encrypt(original)
        with open(f"{data2zip}.ezip", "wb") as encrypted_file:
            encrypted_file.write(encrypted)
        # delete zip file
        os.remove(f"{data2zip}.zip")
    else:
        # zip folder while preserving directory structure but split into multiple files of max size split_size_bytes
        with SplitFileWriter(f"{data2zip}.zip.", split_size_bytes) as sub_zip:
            with zipfile.ZipFile(sub_zip, "w") as fullzip:
                rootlen = (
                    len(data2zip) + 1
                )  # get number of characters to remove from each file path
                for folder, subfolders, files in os.walk(
                    f"{data2zip}"
                ):  # walk through folders
                    for file in files:
                        fn = os.path.join(folder, file)
                        fullzip.write(
                            fn, fn[rootlen:], compress_type=zipfile.ZIP_DEFLATED
                        )
                    for subfolder in subfolders:
                        fn = os.path.join(folder, subfolder)

        # encrypt each split file
        split_files = list(
            filter(lambda x: x.startswith(f"{data2zip}.zip."), os.listdir())
        )
        for split_file in split_files:
            with open(split_file, "rb") as file:
                original = file.read()
            encrypted = fernet.encrypt(original)
            with open(split_file.replace("zip", "ezip"), "wb") as encrypted_file:
                encrypted_file.write(encrypted)
            os.remove(split_file)
