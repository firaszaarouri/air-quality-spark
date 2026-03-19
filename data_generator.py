"""
data_generator.py
─────────────────────────────────────────────────────────────────────────────
Generates synthetic sensor.community-format JSON data for testing the
Air Quality Spark pipeline without needing real sensor network data.

Realism principles:
  - Sensors are placed at real cities (lat/lon + country)
  - PM10/PM2.5 values follow realistic distributions per region
    (Asia/India: high pollution, Scandinavia: low, Europe: moderate)
  - Day-to-day variation adds temporal noise
  - Multi-day trends simulate real pollution events
  - JSON structure exactly matches the sensor.community API format

Usage:
    python data_generator.py          # generates data/ folder
    python data_generator.py --days 7 # custom number of days
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import random
import argparse
import numpy as np
from datetime import datetime, timedelta, date

import config

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Real city locations with pollution profile ────────────────────────────────
# Format: (city, country_alpha2, lat, lon, base_pm10, base_pm25)
# base values represent typical PM10/PM2.5 µg/m³ for each region
SENSOR_LOCATIONS = [
    # Europe — low to moderate pollution
    ("Paris",       "FR",  48.86,  2.35,   25,  15),
    ("Lyon",        "FR",  45.75,  4.83,   20,  12),
    ("Marseille",   "FR",  43.30,  5.37,   22,  13),
    ("Berlin",      "DE",  52.52, 13.40,   20,  12),
    ("Munich",      "DE",  48.14, 11.58,   18,  10),
    ("Hamburg",     "DE",  53.55,  9.99,   19,  11),
    ("London",      "GB",  51.51, -0.12,   22,  14),
    ("Manchester",  "GB",  53.48, -2.24,   21,  13),
    ("Madrid",      "ES",  40.42, -3.70,   28,  16),
    ("Barcelona",   "ES",  41.39,  2.15,   24,  14),
    ("Rome",        "IT",  41.90, 12.50,   30,  18),
    ("Milan",       "IT",  45.46,  9.19,   45,  28),
    ("Amsterdam",   "NL",  52.37,  4.90,   18,  11),
    ("Brussels",    "BE",  50.85,  4.35,   22,  13),
    ("Vienna",      "AT",  48.21, 16.37,   20,  12),
    ("Prague",      "CZ",  50.08, 14.44,   28,  17),
    ("Warsaw",      "PL",  52.23, 21.01,   55,  35),
    ("Krakow",      "PL",  50.06, 19.94,   70,  48),
    ("Budapest",    "HU",  47.50, 19.04,   35,  22),
    ("Bucharest",   "RO",  44.43, 26.10,   40,  26),
    ("Athens",      "GR",  37.98, 23.73,   32,  20),
    ("Stockholm",   "SE",  59.33, 18.07,    8,   5),
    ("Oslo",        "NO",  59.91, 10.75,    7,   4),
    ("Helsinki",    "FI",  60.17, 24.94,    8,   5),
    ("Copenhagen",  "DK",  55.68, 12.57,   10,   6),
    ("Zurich",      "CH",  47.38,  8.54,   12,   7),
    ("Lisbon",      "PT",  38.72, -9.14,   18,  11),
    ("Dublin",      "IE",  53.33, -6.25,   12,   7),
    ("Kyiv",        "UA",  50.45, 30.52,   40,  25),
    ("Lviv",        "UA",  49.84, 24.03,   35,  22),

    # Asia — high pollution
    ("Delhi",       "IN",  28.70, 77.10,  180, 120),
    ("Mumbai",      "IN",  19.08, 72.88,   90,  60),
    ("Kolkata",     "IN",  22.57, 88.36,  140,  95),
    ("Chennai",     "IN",  13.08, 80.27,   70,  45),
    ("Bangalore",   "IN",  12.97, 77.59,   55,  35),
    ("Beijing",     "CN",  39.91, 116.40,  95,  65),
    ("Shanghai",    "CN",  31.23, 121.47,  70,  48),
    ("Guangzhou",   "CN",  23.13, 113.26,  80,  55),
    ("Chengdu",     "CN",  30.57, 104.07,  88,  60),
    ("Seoul",       "KR",  37.57, 126.98,  45,  28),
    ("Busan",       "KR",  35.10, 129.04,  38,  24),
    ("Tokyo",       "JP",  35.69, 139.69,  20,  12),
    ("Osaka",       "JP",  34.69, 135.50,  18,  11),
    ("Bangkok",     "TH",  13.75, 100.52,  65,  42),
    ("Manila",      "PH",  14.60, 120.98,  55,  35),
    ("Jakarta",     "ID",  -6.21, 106.85,  80,  55),
    ("Hanoi",       "VN",  21.03, 105.85,  75,  50),
    ("Karachi",     "PK",  24.86, 67.01,  120,  85),
    ("Lahore",      "PK",  31.55, 74.34,  150, 105),
    ("Dhaka",       "BD",  23.81, 90.41,  160, 110),
    ("Riyadh",      "SA",  24.69, 46.72,   35,  20),
    ("Dubai",       "AE",  25.20, 55.27,   30,  18),

    # Americas
    ("New York",    "US",  40.71, -74.01,  20,  12),
    ("Los Angeles", "US",  34.05, -118.24, 30,  18),
    ("Chicago",     "US",  41.88, -87.63,  22,  13),
    ("Houston",     "US",  29.76, -95.37,  25,  15),
    ("Toronto",     "CA",  43.65, -79.38,  15,   9),
    ("Montreal",    "CA",  45.50, -73.57,  14,   8),
    ("Vancouver",   "CA",  49.25, -123.12,  8,   5),
    ("São Paulo",   "BR", -23.55, -46.63,  35,  22),
    ("Rio",         "BR", -22.91, -43.17,  28,  18),
    ("Buenos Aires","AR", -34.60, -58.38,  22,  14),
    ("Lima",        "PE", -12.05, -77.04,  55,  35),
    ("Bogotá",      "CO",   4.71, -74.07,  40,  25),
    ("Mexico City", "MX",  19.43, -99.13,  60,  40),

    # Africa & Middle East
    ("Cairo",       "EG",  30.06, 31.25,   90,  60),
    ("Lagos",       "NG",   6.52,  3.38,   75,  50),
    ("Nairobi",     "KE",  -1.29, 36.82,   30,  18),
    ("Cape Town",   "ZA", -33.93, 18.42,   15,   9),
    ("Johannesburg","ZA", -26.20, 28.03,   35,  22),
    ("Casablanca",  "MA",  33.59, -7.62,   28,  17),
    ("Accra",       "GH",   5.56, -0.20,   40,  25),
    ("Addis Ababa", "ET",   9.03, 38.74,   45,  30),
    ("Dakar",       "SN",  14.72, -17.47,  35,  22),
    ("Tunis",       "TN",  36.82, 10.17,   25,  15),
    ("Benin City",  "BJ",   6.34,  2.32,   38,  24),

    # Oceania & other
    ("Sydney",      "AU", -33.87, 151.21,  10,   6),
    ("Melbourne",   "AU", -37.81, 144.96,   9,   5),
    ("Auckland",    "NZ", -36.86, 174.77,   7,   4),
    ("Taipei",      "TW",  25.05, 121.56,  38,  24),
    ("Singapore",   "SG",   1.35, 103.82,  25,  15),
    ("Colombo",     "LK",   6.93, 79.85,   45,  28),
    ("Almaty",      "KZ",  43.25, 76.95,   55,  35),
    ("Tashkent",    "UZ",  41.30, 69.24,   60,  40),
    ("Baku",        "AZ",  40.41, 49.87,   40,  25),
    ("Tbilisi",     "GE",  41.69, 44.83,   30,  18),
    ("Minsk",       "BY",  53.90, 27.57,   30,  19),
    ("Vilnius",     "LT",  54.69, 25.28,   15,   9),
    ("Riga",        "LV",  56.95, 24.11,   14,   8),
    ("Tallinn",     "EE",  59.44, 24.75,   10,   6),
    ("Sarajevo",    "BA",  43.85, 18.36,   60,  40),
    ("Skopje",      "MK",  41.99, 21.43,   65,  42),
    ("Sofia",       "BG",  42.70, 23.32,   50,  32),
    ("Chisinau",    "MD",  47.00, 28.86,   30,  19),
    ("Yerevan",     "AM",  40.18, 44.51,   35,  22),
]


def pm_to_aqi(pm10: float, pm25: float) -> int:
    """Convert PM10 and PM2.5 values to AQI (1–10 scale, European CAQI)."""
    def lookup(value, thresholds):
        for i, t in enumerate(thresholds):
            if value <= t:
                return i + 1
        return 10

    aqi1 = lookup(pm10, config.PM10_THRESHOLDS)
    aqi2 = lookup(pm25, config.PM25_THRESHOLDS)
    return max(aqi1, aqi2)


def generate_sensor_json(sensor_id: int, location: tuple, day: date,
                          day_offset: float) -> list:
    """
    Generate realistic sensor readings for one sensor on one day.

    day_offset simulates multi-day pollution trend:
      positive = worsening trend, negative = improving trend
    """
    city, country, lat, lon, base_pm10, base_pm25 = location

    readings = []
    for hour in [7, 13, 20]:   # morning, midday, evening readings
        # Add realistic noise + daily trend
        pm10 = max(0.5, base_pm10 * (1 + day_offset) + np.random.normal(0, base_pm10 * 0.15))
        pm25 = max(0.3, base_pm25 * (1 + day_offset) + np.random.normal(0, base_pm25 * 0.15))

        # Weekend effect: slightly lower (less traffic)
        if day.weekday() >= 5:
            pm10 *= 0.85
            pm25 *= 0.85

        timestamp = datetime.combine(day, datetime.min.time()).replace(hour=hour)

        reading = {
            "id": sensor_id * 100 + hour,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "location": {
                "id": sensor_id,
                "latitude": str(round(lat + np.random.uniform(-0.01, 0.01), 6)),
                "longitude": str(round(lon + np.random.uniform(-0.01, 0.01), 6)),
                "country": country,
                "city": city
            },
            "sensor": {
                "id": sensor_id,
                "sensor_type": {"name": "SDS011"}
            },
            "sensordatavalues": [
                {"value_type": "P1", "value": str(round(pm10, 2))},
                {"value_type": "P2", "value": str(round(pm25, 2))}
            ]
        }
        readings.append(reading)

    return readings


def generate_dataset(output_dir: str = 'data/', n_days: int = 22):
    """
    Generate a full synthetic dataset of sensor readings.

    Creates one JSON file per day, each containing all sensor readings for that day.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select sensors (can be more than SENSOR_LOCATIONS by resampling)
    sensors = SENSOR_LOCATIONS[:config.N_SENSORS] if config.N_SENSORS <= len(SENSOR_LOCATIONS) \
              else SENSOR_LOCATIONS * (config.N_SENSORS // len(SENSOR_LOCATIONS) + 1)
    sensors = sensors[:config.N_SENSORS]

    # Generate per-sensor multi-day trend (-0.3 to +0.3)
    trends = {i: np.random.uniform(-0.3, 0.3) for i in range(len(sensors))}

    # Start date
    start_date = datetime.today().date() - timedelta(days=n_days - 1)

    print(f"Generating {n_days} days of data for {len(sensors)} sensors...")

    for day_idx in range(n_days):
        current_day   = start_date + timedelta(days=day_idx)
        day_readings  = []

        for sensor_idx, location in enumerate(sensors):
            # Progressive trend over days
            day_offset = trends[sensor_idx] * (day_idx / n_days)
            readings   = generate_sensor_json(sensor_idx + 1, location,
                                              current_day, day_offset)
            day_readings.extend(readings)

        # Write one JSON file per day
        filename = os.path.join(output_dir, f"sensors_{current_day.strftime('%Y%m%d')}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            for record in day_readings:
                f.write(json.dumps(record) + '\n')

        print(f"  [{day_idx+1:2d}/{n_days}] {filename} — {len(day_readings)} readings")

    print(f"\nDataset generated: {n_days} files in {output_dir}")
    print(f"Total readings: {n_days * len(sensors) * config.READINGS_PER_DAY:,}")


def generate_country_codes(output_dir: str = 'data/'):
    """Generate a country codes JSON file mapping alpha2 → alpha3 and english name."""
    os.makedirs(output_dir, exist_ok=True)
    codes = [
        {"alpha2Code": "FR", "alpha3Code": "FRA", "englishShortName": "France"},
        {"alpha2Code": "DE", "alpha3Code": "DEU", "englishShortName": "Germany"},
        {"alpha2Code": "GB", "alpha3Code": "GBR", "englishShortName": "United Kingdom"},
        {"alpha2Code": "ES", "alpha3Code": "ESP", "englishShortName": "Spain"},
        {"alpha2Code": "IT", "alpha3Code": "ITA", "englishShortName": "Italy"},
        {"alpha2Code": "NL", "alpha3Code": "NLD", "englishShortName": "Netherlands"},
        {"alpha2Code": "BE", "alpha3Code": "BEL", "englishShortName": "Belgium"},
        {"alpha2Code": "AT", "alpha3Code": "AUT", "englishShortName": "Austria"},
        {"alpha2Code": "CZ", "alpha3Code": "CZE", "englishShortName": "Czech Republic"},
        {"alpha2Code": "PL", "alpha3Code": "POL", "englishShortName": "Poland"},
        {"alpha2Code": "HU", "alpha3Code": "HUN", "englishShortName": "Hungary"},
        {"alpha2Code": "RO", "alpha3Code": "ROU", "englishShortName": "Romania"},
        {"alpha2Code": "GR", "alpha3Code": "GRC", "englishShortName": "Greece"},
        {"alpha2Code": "SE", "alpha3Code": "SWE", "englishShortName": "Sweden"},
        {"alpha2Code": "NO", "alpha3Code": "NOR", "englishShortName": "Norway"},
        {"alpha2Code": "FI", "alpha3Code": "FIN", "englishShortName": "Finland"},
        {"alpha2Code": "DK", "alpha3Code": "DNK", "englishShortName": "Denmark"},
        {"alpha2Code": "CH", "alpha3Code": "CHE", "englishShortName": "Switzerland"},
        {"alpha2Code": "PT", "alpha3Code": "PRT", "englishShortName": "Portugal"},
        {"alpha2Code": "IE", "alpha3Code": "IRL", "englishShortName": "Ireland"},
        {"alpha2Code": "UA", "alpha3Code": "UKR", "englishShortName": "Ukraine"},
        {"alpha2Code": "IN", "alpha3Code": "IND", "englishShortName": "India"},
        {"alpha2Code": "CN", "alpha3Code": "CHN", "englishShortName": "China"},
        {"alpha2Code": "KR", "alpha3Code": "KOR", "englishShortName": "South Korea"},
        {"alpha2Code": "JP", "alpha3Code": "JPN", "englishShortName": "Japan"},
        {"alpha2Code": "TH", "alpha3Code": "THA", "englishShortName": "Thailand"},
        {"alpha2Code": "PH", "alpha3Code": "PHL", "englishShortName": "Philippines"},
        {"alpha2Code": "ID", "alpha3Code": "IDN", "englishShortName": "Indonesia"},
        {"alpha2Code": "VN", "alpha3Code": "VNM", "englishShortName": "Vietnam"},
        {"alpha2Code": "PK", "alpha3Code": "PAK", "englishShortName": "Pakistan"},
        {"alpha2Code": "BD", "alpha3Code": "BGD", "englishShortName": "Bangladesh"},
        {"alpha2Code": "SA", "alpha3Code": "SAU", "englishShortName": "Saudi Arabia"},
        {"alpha2Code": "AE", "alpha3Code": "ARE", "englishShortName": "United Arab Emirates"},
        {"alpha2Code": "US", "alpha3Code": "USA", "englishShortName": "United States"},
        {"alpha2Code": "CA", "alpha3Code": "CAN", "englishShortName": "Canada"},
        {"alpha2Code": "BR", "alpha3Code": "BRA", "englishShortName": "Brazil"},
        {"alpha2Code": "AR", "alpha3Code": "ARG", "englishShortName": "Argentina"},
        {"alpha2Code": "PE", "alpha3Code": "PER", "englishShortName": "Peru"},
        {"alpha2Code": "CO", "alpha3Code": "COL", "englishShortName": "Colombia"},
        {"alpha2Code": "MX", "alpha3Code": "MEX", "englishShortName": "Mexico"},
        {"alpha2Code": "EG", "alpha3Code": "EGY", "englishShortName": "Egypt"},
        {"alpha2Code": "NG", "alpha3Code": "NGA", "englishShortName": "Nigeria"},
        {"alpha2Code": "KE", "alpha3Code": "KEN", "englishShortName": "Kenya"},
        {"alpha2Code": "ZA", "alpha3Code": "ZAF", "englishShortName": "South Africa"},
        {"alpha2Code": "MA", "alpha3Code": "MAR", "englishShortName": "Morocco"},
        {"alpha2Code": "GH", "alpha3Code": "GHA", "englishShortName": "Ghana"},
        {"alpha2Code": "ET", "alpha3Code": "ETH", "englishShortName": "Ethiopia"},
        {"alpha2Code": "SN", "alpha3Code": "SEN", "englishShortName": "Senegal"},
        {"alpha2Code": "TN", "alpha3Code": "TUN", "englishShortName": "Tunisia"},
        {"alpha2Code": "BJ", "alpha3Code": "BEN", "englishShortName": "Benin"},
        {"alpha2Code": "AU", "alpha3Code": "AUS", "englishShortName": "Australia"},
        {"alpha2Code": "NZ", "alpha3Code": "NZL", "englishShortName": "New Zealand"},
        {"alpha2Code": "TW", "alpha3Code": "TWN", "englishShortName": "Taiwan"},
        {"alpha2Code": "SG", "alpha3Code": "SGP", "englishShortName": "Singapore"},
        {"alpha2Code": "LK", "alpha3Code": "LKA", "englishShortName": "Sri Lanka"},
        {"alpha2Code": "KZ", "alpha3Code": "KAZ", "englishShortName": "Kazakhstan"},
        {"alpha2Code": "UZ", "alpha3Code": "UZB", "englishShortName": "Uzbekistan"},
        {"alpha2Code": "AZ", "alpha3Code": "AZE", "englishShortName": "Azerbaijan"},
        {"alpha2Code": "GE", "alpha3Code": "GEO", "englishShortName": "Georgia"},
        {"alpha2Code": "BY", "alpha3Code": "BLR", "englishShortName": "Belarus"},
        {"alpha2Code": "LT", "alpha3Code": "LTU", "englishShortName": "Lithuania"},
        {"alpha2Code": "LV", "alpha3Code": "LVA", "englishShortName": "Latvia"},
        {"alpha2Code": "EE", "alpha3Code": "EST", "englishShortName": "Estonia"},
        {"alpha2Code": "BA", "alpha3Code": "BIH", "englishShortName": "Bosnia and Herzegovina"},
        {"alpha2Code": "MK", "alpha3Code": "MKD", "englishShortName": "North Macedonia"},
        {"alpha2Code": "BG", "alpha3Code": "BGR", "englishShortName": "Bulgaria"},
        {"alpha2Code": "MD", "alpha3Code": "MDA", "englishShortName": "Moldova"},
        {"alpha2Code": "AM", "alpha3Code": "ARM", "englishShortName": "Armenia"},
    ]
    path = os.path.join(output_dir, 'country_codes.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(codes, f, indent=2, ensure_ascii=False)
    print(f"Country codes written: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic air quality data')
    parser.add_argument('--days',    type=int, default=config.N_DAYS,    help='Number of days')
    parser.add_argument('--sensors', type=int, default=config.N_SENSORS, help='Number of sensors')
    parser.add_argument('--output',  type=str, default=config.DATA_DIR,  help='Output directory')
    args = parser.parse_args()

    config.N_DAYS    = args.days
    config.N_SENSORS = args.sensors

    generate_country_codes(args.output)
    generate_dataset(args.output, args.days)
