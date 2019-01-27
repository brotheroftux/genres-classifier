from . import generate_data as gen
import json, codecs
import os
import os.path as path

__location__ = path.realpath(
    path.join(os.getcwd(), path.dirname(__file__)))

ENTRIES = 1000
FILENAME = path.join(__location__, 'data.json')

def get_data ():
  global ENTRIES, FILENAME

  if not path.exists(FILENAME):
    print('\033[93m', 'No dataset found, generating.', '\033[0m')
    print('\033[1m', 'Sample size: {}'.format(ENTRIES), '\033[0m')

    users, labels = gen.generate_dataset(ENTRIES)

    json_pre_serialize = {
      'users': users,
      'labels': labels
    }

    json.dump(json_pre_serialize, codecs.open(FILENAME, 'w', encoding='utf-8'))

    return (users, labels)
  else:
    print('\033[94m', 'Dataset found, loading.', '\033[0m')

    data = json.load(codecs.open(FILENAME, encoding='utf-8'))

    users = data['users']
    labels = data['labels']

    return (users, labels)