from __future__ import annotations

import math

import numpy as np

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _validate_geodetic(latitude_deg: float, longitude_deg: float, altitude_m: float) -> None:
    if not all(math.isfinite(float(v)) for v in (latitude_deg, longitude_deg, altitude_m)):
        raise ValueError("Geodetic coordinates must be finite")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude must be in [-90, 90]")
    if not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("longitude must be in [-180, 180]")


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float = 0.0) -> np.ndarray:
    _validate_geodetic(latitude_deg, longitude_deg, altitude_m)
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)
    sin_lon = math.sin(longitude)
    cos_lon = math.cos(longitude)
    prime_vertical = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (prime_vertical + altitude_m) * cos_lat * cos_lon
    y = (prime_vertical + altitude_m) * cos_lat * sin_lon
    z = (prime_vertical * (1.0 - _WGS84_E2) + altitude_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


class LocalENUFrame:
    def __init__(self, latitude_deg: float, longitude_deg: float, altitude_m: float = 0.0) -> None:
        _validate_geodetic(latitude_deg, longitude_deg, altitude_m)
        self.latitude_deg = float(latitude_deg)
        self.longitude_deg = float(longitude_deg)
        self.altitude_m = float(altitude_m)
        self.origin_ecef = geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)
        latitude = math.radians(latitude_deg)
        longitude = math.radians(longitude_deg)
        sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
        sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
        self.R_ecef_to_enu = np.array(
            [
                [-sin_lon, cos_lon, 0.0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ],
            dtype=np.float64,
        )

    def geodetic_to_enu(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float = 0.0,
    ) -> tuple[float, float, float]:
        point_ecef = geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)
        enu = self.R_ecef_to_enu @ (point_ecef - self.origin_ecef)
        return float(enu[0]), float(enu[1]), float(enu[2])
