import generate_data as gen
import json, codecs
import os.path as path

ENTRIES = 1000
FILENAME = 'data.json'

def get_data ():
  global ENTRIES, FILENAME

  if not path.exists(FILENAME):
    users, labels = gen.generate_dataset(ENTRIES)

    json_pre_serialize = {
      'users': users,
      'labels': labels
    }

    json.dump(json_pre_serialize, codecs.open(FILENAME, 'w', encoding='utf-8'))

    return (users, labels)
  else:
    data = json.load(codecs.open(FILENAME, encoding='utf-8'))

    users = data['users']
    labels = data['labels']

    return (users, labels)