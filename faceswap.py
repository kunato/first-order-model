from animate import normalize_kp
from modules.keypoint_detector import KPDetector
from modules.generator import OcclusionAwareGenerator
from sync_batchnorm import DataParallelWithCallback
import torch
from skimage.transform import resize
import numpy as np
import imageio
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
import os
import matplotlib
import cv2
matplotlib.use('Agg')


def load_checkpoints(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.cuda()

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    # generator = DataParallelWithCallback(generator)
    # kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True):
    # from face_detection.face_detector import MTCNNFaceDetector
    # from face_detection.transformer.landmarks_alignment import get_src_landmarks, get_tar_landmarks, landmarks_match_mtcnn
    from face_detection.transformer.color_correction import adain
    # import keras.backend as K
    import face_alignment
    # K.set_learning_phase(0)
    # fd = MTCNNFaceDetector(sess=K.get_session())
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, device='cuda', flip_input=True)

    def findCenter(img):
        th, threshed = cv2.threshold(
            (img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(
            threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(cnts[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2).cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(
            np.float32)).permute(0, 4, 1, 2, 3).cuda()
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame_np = driving_video[frame_idx]
            preds = fa.get_landmarks(
                (driving_frame_np * 255).astype(np.uint8))[-1]
            ori_mask = np.zeros(
                (driving_frame_np.shape[0], driving_frame_np.shape[1]))
            pnts = [(preds[i, 0], preds[i, 1]) for i in range(1, 42)]
            hull = cv2.convexHull(np.array(pnts)).astype(np.int32)
            ori_mask = cv2.drawContours(ori_mask, [hull], 0, 255, -1)
            ori_mask = cv2.erode(ori_mask, np.ones(
                (21, 21), np.uint8), iterations=1)
            # ori_mask = cv2.GaussianBlur(ori_mask, (7,7), 0)
            ori_mask = ori_mask[:, :].astype(np.float32) / 255.

            # get src/tar landmark

            # faces, pnts = fd.detect_face((driving_frame_np * 255).astype(np.uint8))
            # x0, y1, x1, y0, conf_score = faces[0]
            # lms = pnts[:,0:1]
            # src_landmarks = get_src_landmarks(x0, x1, y0, y1, lms)

            driving_frame = driving[:, :, frame_idx]

            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            r_rgb = np.transpose(
                out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

            preds = fa.get_landmarks((r_rgb*255).astype(np.uint8))[-1]
            mask = np.zeros(
                (driving_frame_np.shape[0], driving_frame_np.shape[1]), dtype=np.uint8)
            pnts = [(preds[i, 0], preds[i, 1]) for i in range(17, 68)]
            hull = cv2.convexHull(np.array(pnts)).astype(np.int32)
            mask = cv2.drawContours(mask, [hull], -1, 255, -1)
            mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
            # mask = cv2.GaussianBlur(mask, (7,7), 0)
            mask = mask[:, :].astype(np.float32) / 255.
        
            # faces, pnts = fd.detect_face((r_rgb * 255).astype(np.uint8))
            # x0, y1, x1, y0, conf_score = faces[0]
            # lms = pnts[:,0:1]
            # tar_landmarks = get_src_landmarks(x0, x1, y0, y1, lms)

            # r_rgb = landmarks_match_mtcnn(r_rgb, tar_landmarks, src_landmarks)
            # mask = landmarks_match_mtcnn(mask, tar_landmarks, src_landmarks)
            # cv2.imwrite('rgb_r.png', (r_rgb * 255).astype(np.uint8))

            mask = cv2.bitwise_and(mask, ori_mask)
            # cv2.imwrite('mask.png', (mask * 255).astype(np.uint8))
            # cv2.imwrite('driving_frame_np.png', (driving_frame_np * 255).astype(np.uint8))

            # r_rgb_crop = r_rgb[int(x0):int(x1),int(y0):int(y1),:]
            # r_a = np.zeros((r_rgb_crop.shape[0], r_rgb_crop.shape[1]))
            # roi_x, roi_y = int(r_rgb_crop.shape[0]*(1-0.9)), int(r_rgb_crop.shape[1]*(1-0.9))
            # r_a[roi_x:-roi_x, roi_y:-roi_y] = 1.
            # cv2.imwrite('mask.png', (r_a * 255).astype(np.uint8))
            # rev_aligned_det_face_im_rgb = r_rgb_crop
            # rev_aligned_mask = r_a
            # # rev_aligned_det_face_im_rgb = landmarks_match_mtcnn(r_rgb_crop, tar_landmarks, src_landmarks)
            # # rev_aligned_mask = landmarks_match_mtcnn(r_a, tar_landmarks, src_landmarks)
            # rev_aligned_mask = rev_aligned_mask[:,:, None]
            # cv2.imwrite('r_rgb.png', (r_rgb_crop * 255).astype(np.uint8))
            # # cv2.imwrite('rev_aligned_det_face_im_rgb.png', (rev_aligned_det_face_im_rgb * 255).astype(np.uint8))
            # cv2.imwrite('driving.png', (det_face_im * 255).astype(np.uint8))
            # print('shape : ', rev_aligned_mask.shape, det_face_im.shape, rev_aligned_det_face_im_rgb.shape)
            # # result = np.zeros_like(det_face_im)
            # result = rev_aligned_mask*rev_aligned_det_face_im_rgb + (1-rev_aligned_mask)*det_face_im
            # cv2.imwrite('result.png', (result * 255).astype(np.uint8))

            # center = findCenter(mask)

            # print(center)
            # result = cv2.seamlessClone((r_rgb * 255).astype(np.uint8), (driving_frame_np * 255).astype(np.uint8), (mask* 255).astype(np.uint8), center, cv2.NORMAL_CLONE).astype(np.float32) / 255.
            # r_rgb = color_hist_match((r_rgb * 255).astype(np.uint8), (driving_frame_np * 255).astype(np.uint8)) / 255.

            th, threshed = cv2.threshold(
                (mask * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
            threshed = cv2.dilate(threshed, np.ones(
                (15, 15), dtype=np.float32), iterations=3)
            cnts, _ = cv2.findContours(
                threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_bbox = cv2.boundingRect(cnts[0])

            # print(mask_bbox)
            r_rgb[mask_bbox[1]:mask_bbox[1] + mask_bbox[3], mask_bbox[0]: mask_bbox[0] + mask_bbox[2], :] = adain((r_rgb[mask_bbox[1]:mask_bbox[1] + mask_bbox[3], mask_bbox[0]: mask_bbox[0] + mask_bbox[2], :] * 255).astype(
                np.uint8), (driving_frame_np[mask_bbox[1]:mask_bbox[1] + mask_bbox[3], mask_bbox[0]: mask_bbox[0] + mask_bbox[2], :] * 255).astype(np.uint8), color_space='xyz') / 255.
            # cv2.imwrite('r_rgb.png', (r_rgb * 255).astype(np.uint8))
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask = mask[:, :, None]

            result = r_rgb * mask + (1 - mask) * driving_frame_np
            # cv2.imwrite('result.png', (result * 255).astype(np.uint8))
            predictions.append(result)

    return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar',
                        help="path to checkpoint to restore")

    parser.add_argument(
        "--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument(
        "--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument(
        "--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    reader.close()
    driving_video = imageio.mimread(opt.driving_video, memtest=False)

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3]
                     for frame in driving_video]
    generator, kp_detector = load_checkpoints(
        config_path=opt.config, checkpoint_path=opt.checkpoint)
    
    source_image_name = opt.source_image.split('/')[-1].split('.')[-2]
    driving_video_name = opt.driving_video.split('/')[-1].split('.')[-2]


    predictions = make_animation(source_image, driving_video, generator,
                                 kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale)
    imageio.mimsave(f'{driving_video_name}_{source_image_name}.mp4', predictions, fps=fps)
