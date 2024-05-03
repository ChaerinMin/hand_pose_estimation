import os
import re
import sys
from urllib import request as urlrequest

def download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    """Download url and write it to dst_file_path. Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    # url = url + "?dl=1" if "dropbox" in url else url
    req = urlrequest.Request(url)
    response = urlrequest.urlopen(req)
    total_size = response.info().get("Content-Length")
    if total_size is None:
        raise ValueError("Cannot determine size of download from {}".format(url))
    total_size = int(total_size.strip())
    bytes_so_far = 0

    with open(dst_file_path, "wb") as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break

            if progress_hook:
                progress_hook(bytes_so_far, total_size)

            f.write(chunk)
    return bytes_so_far