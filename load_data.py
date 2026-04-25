import pandas as pd
import psycopg2

DB_HOST     = "127.0.0.1"
DB_PORT     = 5432
DB_NAME     = "spotify"
DB_USER     = "root"
DB_PASSWORD = "root"

conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
)
cur = conn.cursor()

print("Loading artists...")
artists = pd.read_csv("artists.csv", encoding="utf-8")
for _, row in artists.iterrows():
    cur.execute(
        "INSERT INTO artists (id, name, followers, popularity, genres, main_genre) "
        "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING;",
        (row.get("id"), row.get("name"), row.get("followers"),
         row.get("popularity"), row.get("genres"), row.get("main_genre"))
    )
print(f"  {len(artists)} artists loaded.")

print("Loading songs...")
songs = pd.read_csv("songs.csv", encoding="utf-8")
for _, row in songs.iterrows():
    cur.execute(
        "INSERT INTO songs (id, name, album_name, artists, danceability, energy, key, "
        "loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, "
        "tempo, duration_ms, lyrics, year, genre, popularity, total_artist_followers, "
        "avg_artist_popularity, artist_ids, niche_genres) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (id) DO NOTHING;",
        tuple(None if pd.isna(v) else v for v in [
            row.get("id"), row.get("name"), row.get("album_name"), row.get("artists"),
            row.get("danceability"), row.get("energy"), row.get("key"), row.get("loudness"),
            row.get("mode"), row.get("speechiness"), row.get("acousticness"),
            row.get("instrumentalness"), row.get("liveness"), row.get("valence"),
            row.get("tempo"), row.get("duration_ms"), row.get("lyrics"), row.get("year"),
            row.get("genre"), row.get("popularity"), row.get("total_artist_followers"),
            row.get("avg_artist_popularity"), row.get("artist_ids"), row.get("niche_genres")
        ])
    )
print(f"  {len(songs)} songs loaded.")

conn.commit()
cur.close()
conn.close()
print("Done.")
