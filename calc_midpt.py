import nvector as nv
points = nv.GeoPoint(
    latitude=[-90, -77.8564],
    longitude=[166.6881, 166.6881], degrees=True,
)
nvectors = points.to_nvector()
n_EM_E = nvectors.mean()
g_EM_E = n_EM_E.to_geo_point()
lat, lon = g_EM_E.latitude_deg, g_EM_E.longitude_deg
print('Midpt. Lat: %2.2f, Lon: %2.2f' % (lat, lon))
