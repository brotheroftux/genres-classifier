import random

genre_ranges = {
  'K-Pop': (
    (0, 12), (77, 79)
  ),
  'Drum and Bass': (
    (12, 15), (79, 85)
  ),
  'Symphonic Metal': (
    (85, 100), (15, 17), (44, 47)
  ),
  'Trance': (
    (17, 44),
  ),
  'Progressive Rock': (
    (47, 77),
  )
}

def generate_user ():
  global genre_ranges
  genres_count = random.randint(1, 5)
  genres = random.sample(genre_ranges.keys(), genres_count)

  tracks = []

  for i in range(random.randrange(7, 11)):
    genre = random.choice(genres)
    
    ranges = genre_ranges[genre] 
    start, end = random.choice(ranges)
    
    track_id = random.randrange(start, end)

    tracks.append(track_id)

  return (tracks, genres)

def generate_dataset (rows):
  users = []
  labels = []

  for i in range(rows):
    user, label = generate_user()

    users.append(user)
    labels.append(label)
  
  return (users, labels)
