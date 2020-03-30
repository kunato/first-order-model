import cv2
from moviepy.editor import ImageSequenceClip


def main():
    result_video = cv2.VideoCapture('result.mp4')
    full_video = cv2.VideoCapture('xOZ.mp4')
    fps = result_video.get(cv2.CAP_PROP_FPS)

    full_success, full_image = full_video.read()
    result_sucess, result_image = result_video.read()
    crop_size = (288, 289)
    left_top = (765, 99)

    edited_frame = []

    while full_success and result_sucess:
        result_image = cv2.resize(result_image, crop_size)
        full_image[left_top[1]: left_top[1] + crop_size[1], left_top[0]: left_top[0] + crop_size[0], :] = result_image
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        edited_frame.append(full_image)
        full_success, full_image = full_video.read()
        result_sucess, result_image = result_video.read()

    clip = ImageSequenceClip(edited_frame, fps=fps)

    clip.write_videofile('full_edited.mp4')

if __name__ == "__main__":
    main()