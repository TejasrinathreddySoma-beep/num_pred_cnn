
import os
import tarfile
import zipfile
import hashlib
import requests
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/19003721"

DATA_DIR = Path("data") / "iam"
DOWNLOAD_DIR = DATA_DIR / "downloads"

REQUIRED_FILES = [
    "lines.tar",
    "ascii.tar",
    "largeWriterIndependentTextLineRecognitionTask.zip",
]


def verify_file(path, expected_size, expected_checksum):
    """Check downloaded file size and checksum."""
    if not path.exists():
        return False

    if path.stat().st_size != expected_size:
        print(f"Incorrect file size: {path.name}")
        return False

    if expected_checksum:
        algorithm, expected_hash = expected_checksum.split(":", 1)
        hasher = hashlib.new(algorithm)

        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(chunk)

        if hasher.hexdigest() != expected_hash:
            print(f"Checksum failed: {path.name}")
            return False

    return True


def download_file(url, destination, expected_size, checksum):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if verify_file(destination, expected_size, checksum):
        print(f"Already downloaded and verified: {destination.name}")
        return

    # Remove an incomplete or corrupted old file
    if destination.exists():
        print(f"Removing incomplete file: {destination.name}")
        destination.unlink()

    temp_file = destination.with_suffix(destination.suffix + ".part")

    if temp_file.exists():
        temp_file.unlink()

    print(f"\nDownloading: {destination.name}")

    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()

            downloaded = 0

            with open(temp_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        if expected_size:
                            percent = downloaded * 100 / expected_size
                            print(
                                f"\rProgress: {percent:.1f}%",
                                end="",
                                flush=True,
                            )

        print()

        if not verify_file(temp_file, expected_size, checksum):
            raise ValueError(
                f"Downloaded file failed verification: {destination.name}"
            )

        temp_file.replace(destination)
        print(f"Download verified: {destination.name}")

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


def safe_extract_tar(archive, destination):
    destination = destination.resolve()

    print(f"\nChecking TAR archive: {archive.name}")

    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()

        for member in members:
            target = (destination / member.name).resolve()

            if os.path.commonpath(
                    [str(destination), str(target)]
            ) != str(destination):
                raise ValueError(f"Unsafe archive path: {member.name}")

        print(f"Extracting {len(members)} items...")
        tar.extractall(destination)

    print(f"Extracted: {archive.name}")


def safe_extract_zip(archive, destination):
    destination = destination.resolve()

    with zipfile.ZipFile(archive, "r") as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()

            if os.path.commonpath(
                    [str(destination), str(target)]
            ) != str(destination):
                raise ValueError(f"Unsafe archive path: {member.filename}")

        print(f"\nExtracting: {archive.name}")
        zip_file.extractall(destination)

    print(f"Extracted: {archive.name}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to Zenodo...")

    response = requests.get(ZENODO_API, timeout=60)
    response.raise_for_status()

    record = response.json()

    available_files = {
        item["key"]: item
        for item in record.get("files", [])
    }

    print("\nChecking IAM files...")

    for filename in REQUIRED_FILES:
        if filename not in available_files:
            print(f"File not found on Zenodo: {filename}")
            return

    # Download and verify each required file
    for filename in REQUIRED_FILES:
        info = available_files[filename]

        download_file(
            url=info["links"]["self"],
            destination=DOWNLOAD_DIR / filename,
            expected_size=info["size"],
            checksum=info.get("checksum"),
        )

    # Extract only after all files have passed verification
    print("\nAll downloads verified. Starting extraction...")

    for filename in REQUIRED_FILES:
        archive = DOWNLOAD_DIR / filename

        if filename.endswith(".tar"):
            safe_extract_tar(archive, DATA_DIR)

        elif filename.endswith(".zip"):
            safe_extract_zip(archive, DATA_DIR)

    print("\n====================================")
    print("IAM DOWNLOAD AND EXTRACTION COMPLETE")
    print("====================================")
    print(f"Dataset folder: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
