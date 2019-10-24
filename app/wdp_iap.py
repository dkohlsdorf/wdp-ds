import requests
import sys

from jose import jwt 

def validate_assertion(assertion):
    try:
        info = jwt.decode(assertion, certs(), algorithm=['ES256'],audience=audience())
        return info['email'], info['sub']
    except Exception as e:
        print("Failed to validate assertion: {} ".format(e), file=sys.stderr)
        return None, None

def audience():
    global AUDIENCE
    if AUDIENCE is None:
        project_number = get_metadata('numeric-project-id')
        project_id = get_metadata('project-id')
        AUDIENCE = 'projects/{}/apps/{}'.format(
            project_number, project_id
        )
    return AUDIENCE

def get_metadata(item_name):
    endpoint = 'http://metadata.google.internal'
    path     = '/computeMetaData/v1/project/'
    path    += item_name
    response = requests.get(
        '{}{}'.format(endpoint, path),
        headers={'Metadata-Flavor' : 'Google'}
    )
    metadata = response.text
    return metadata

def certs():
    global CERTS
    if CERTS is None:
        response = requests.get(
            'https://www.gstatic.com/iap/verify/public_key'
        )
        CERTS = response.json
    return CERTS