import cv2

def frame_preprocess(path):
    stream = cv2.VideoCapture(path)
    assert stream.isOpened(), 'Cannot capture source'

    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_imgs = []
    im_names = []
    
    frame_num = 0
    for k in range(datalen):
        if k % 3 == 0 or k % 3 == 1 or k % 3 == 2:
            (grabbed, frame) = stream.read()
            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                stream.release()
                break

            # orig_imgs.append(frame[:, :, ::-1])
            orig_imgs.append(frame)
            im_names.append(f'{frame_num:08d}' + '.jpg')
            frame_num += 1

    stream.release()

    print(f'Total number of frames: {frame_num} in {path}')
    return im_names, orig_imgs