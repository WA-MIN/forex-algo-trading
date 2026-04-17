from pathlib import Path
from io import BytesIO
import time
import zipfile

import pandas as pd
import requests
from bs4 import BeautifulSoup

PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
]

YEARS = list(range(2015, 2026))

BASE_URL = "https://www.histdata.com"
PAGE_URL_TEMPLATE = BASE_URL + "/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}"

DATA_DIR = Path("../data")
EXTRACTED_DIR = DATA_DIR / "extracted"
PARQUET_DIR = DATA_DIR / "parquet"

EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}


def get_session(timestamp_utc: pd.Timestamp) -> str:
    """Label rows by broad trading session using UTC time."""
    hour = timestamp_utc.hour

    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 13:
        return "London"
    if 13 <= hour < 17:
        return "Overlap"
    return "New_York"


def get_download_form_values(session: requests.Session, pair: str, year: int) -> dict:
    """Open the yearly HistData page and extract the hidden download form values."""
    url = PAGE_URL_TEMPLATE.format(pair=pair.lower(), year=year)

    response = session.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    form = soup.find("form", {"id": "file_down"})
    if form is None:
        raise ValueError(f"Could not find hidden download form for {pair} {year}")

    payload = {}
    for input_tag in form.find_all("input"):
        name = input_tag.get("name")
        value = input_tag.get("value", "")
        if name:
            payload[name] = value

    required = ["tk", "date", "datemonth", "platform", "timeframe", "fxpair"]
    missing = [key for key in required if key not in payload or payload[key] == ""]
    if missing:
        raise ValueError(f"Missing hidden fields for {pair} {year}: {missing}")

    return payload


def download_zip_bytes(session: requests.Session, pair: str, year: int) -> bytes:
    """Submit the hidden download form and return the ZIP bytes."""
    page_url = PAGE_URL_TEMPLATE.format(pair=pair.lower(), year=year)
    payload = get_download_form_values(session, pair, year)

    response = session.post(
        BASE_URL + "/get.php",
        data=payload,
        headers={
            **HEADERS,
            "Origin": BASE_URL,
            "Referer": page_url,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.content

    if len(data) < 100:
        raise ValueError(f"Downloaded file is too small for {pair} {year}")

    if data[:4] != b"PK\x03\x04":
        sample = data[:200].decode("utf-8", errors="ignore")
        raise ValueError(f"Response is not a ZIP file for {pair} {year}. First bytes: {sample!r}")

    return data


def extract_year_file(zip_bytes: bytes, pair: str, year: int) -> Path:
    """Extract the CSV file from the ZIP and save it as a yearly CSV."""
    pair_dir = EXTRACTED_DIR / pair
    pair_dir.mkdir(parents=True, exist_ok=True)

    output_path = pair_dir / f"{pair}_{year}.csv"

    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

        data_files = [
            name for name in names
            if name.lower().endswith(".csv") or name.lower().endswith(".txt")
        ]

        if not data_files:
            raise ValueError(f"No csv or txt file found inside ZIP for {pair} {year}")

        data_file = data_files[0]

        with zf.open(data_file) as src, open(output_path, "wb") as dst:
            dst.write(src.read())

    return output_path


def download_and_extract_year(session: requests.Session, pair: str, year: int) -> Path:
    """Download yearly file and save the extracted CSV."""
    output_path = EXTRACTED_DIR / pair / f"{pair}_{year}.csv"

    if output_path.exists():
        print(f"Skipping {pair} {year}, extracted CSV already exists")
        return output_path

    print(f"Downloading {pair} {year}")
    zip_bytes = download_zip_bytes(session, pair, year)
    extracted_path = extract_year_file(zip_bytes, pair, year)
    print(f"Saved {extracted_path}")

    time.sleep(1)
    return extracted_path


def load_histdata_file(file_path: Path, pair: str) -> pd.DataFrame:
    """Load one extracted HistData yearly file."""
    df = pd.read_csv(
        file_path,
        sep=";",
        header=None,
        names=["timestamp_est", "open", "high", "low", "close", "volume"],
        dtype={
            "timestamp_est": "string",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
        },
    )

    df["timestamp_est"] = pd.to_datetime(
        df["timestamp_est"],
        format="%Y%m%d %H%M%S",
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp_est"]).copy()

    # HistData timestamps are EST without DST.
    df["timestamp_utc"] = (df["timestamp_est"] + pd.Timedelta(hours=5)).dt.tz_localize("UTC")

    df["pair"] = pair
    df["year"] = df["timestamp_utc"].dt.year
    df["month"] = df["timestamp_utc"].dt.month
    df["session"] = df["timestamp_utc"].apply(get_session)

    df = df[
        [
            "timestamp_est",
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "pair",
            "year",
            "month",
            "session",
        ]
    ].copy()

    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).reset_index(drop=True)
    return df


def process_pair_to_parquet(pair: str) -> None:
    """Combine all yearly CSVs for one pair into one parquet file."""
    pair_dir = EXTRACTED_DIR / pair
    if not pair_dir.exists():
        print(f"No extracted files found for {pair}")
        return

    frames = []

    for year in YEARS:
        file_path = pair_dir / f"{pair}_{year}.csv"
        if not file_path.exists():
            print(f"Missing extracted file for {pair} {year}")
            continue

        try:
            df = load_histdata_file(file_path, pair)
            frames.append(df)
        except Exception as exc:
            print(f"Failed reading {file_path}: {exc}")

    if not frames:
        print(f"No usable data found for {pair}")
        return

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).reset_index(drop=True)

    output_path = PARQUET_DIR / f"{pair}_2015_2025.parquet"
    full_df.to_parquet(output_path, index=False)

    print(f"Saved parquet for {pair}: {len(full_df):,} rows")
    print(f"Path: {output_path}")


def main() -> None:
    session = requests.Session()

    for pair in PAIRS:
        for year in YEARS:
            try:
                download_and_extract_year(session, pair, year)
            except Exception as exc:
                print(f"Failed {pair} {year}: {exc}")

    for pair in PAIRS:
        try:
            process_pair_to_parquet(pair)
        except Exception as exc:
            print(f"Parquet processing failed for {pair}: {exc}")

    print("Done")
    print("Extracted CSV files are in data/extracted/")
    print("Parquet files are in data/parquet/")


if __name__ == "__main__":
    main()
