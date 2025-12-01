import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "rdp_client",
        "sim",
    },
    submod_attrs={
        "rdp_client": [
            "unlock_and_unzip_file",
            "zip_and_lock_folder",
        ],
    },
)

__all__ = ["rdp_client", "sim", "unlock_and_unzip_file", "zip_and_lock_folder"]
