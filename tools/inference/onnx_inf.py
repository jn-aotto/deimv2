"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import time
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, labels, boxes, scores, ratios, paddings, thrh=0.4):
    result_images = []
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        # 设置字体大小（可以根据图片尺寸调整）
        try:
            # 尝试加载字体，如果失败则使用默认字体
            font_size = 40  # 字体大小，可以根据需要调整
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # 如果找不到字体文件，使用默认字体
            font = ImageFont.load_default()

        for lbl, bb in zip(lab, box):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline='red', width=5)  # width=3 表示3像素粗的线条
            # 使用font参数设置文字大小
            draw.text((bb[0], bb[1]), text=str(lbl), fill='blue', font=font)

        result_images.append(im)
    return result_images


def process_image(sess, im_pil, image_path=None, size=640, model_size='s'):
    # Resize image while preserving aspect ratio
    start_time = time.time()
    # 传递原始图片路径作为保存路径
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, size)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if model_size not in ['atto', 'femto', 'pico', 'n'] 
                else T.Lambda(lambda x: x)
        ])
    im_data = transforms(resized_im_pil).unsqueeze(0)
    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
    )

    labels, boxes, scores = output # (1, 300) (1, 300, 4) (1, 300)
    end_time = time.time()
    print(f"推理时间: {end_time - start_time:.4f} 秒")

    #print(labels.shape, boxes.shape, scores.shape)
    # 输出前五项得分最高的类别以及框的位置
    # if len(scores[0]) > 0:
    #     # 获取所有检测的分数并按降序排序
    #     sorted_indices = np.argsort(scores[0])[::-1]  # 降序排序的索引
    #     top_5_indices = sorted_indices[:5]  # 取前5个
        
    #     # print("前五项得分最高的检测结果:")
    #     # for i, idx in enumerate(top_5_indices):
    #     #     if idx < len(scores[0]):  # 确保索引有效
    #     #         label = labels[0][idx]
    #     #         box = boxes[0][idx]
    #     #         score = scores[0][idx]
    #     #         print(f"第{i+1}名 - 类别: {label}, 得分: {score:.4f}, 边界框: {box}")
    # else:
    #     print("未检测到任何对象")
    
    # print(f"\n所有检测结果统计:")
    # print(f"总检测数量: {len(scores[0])}")

    result_images = draw(
        [im_pil], labels, boxes, scores,
        [ratio], [(pad_w, pad_h)]
    )
    # 如果提供了原始图片路径，使用它作为保存路径，否则使用默认路径
    save_path = image_path or 'onnx_result.jpg'
    result_images[0].save(save_path)
    print(f"Image processing complete. Result saved as '{save_path}'.")


def process_video(sess, video_path, size=640, model_size='s'):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('onnx_result.mp4', fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize frame while preserving aspect ratio
        resized_frame_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, size)
        orig_size = torch.tensor([[resized_frame_pil.size[1], resized_frame_pil.size[0]]])

        transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                    if model_size not in ['atto', 'femto', 'pico', 'n'] 
                    else T.Lambda(lambda x: x)
            ])
        im_data = transforms(resized_frame_pil).unsqueeze(0)

        output = sess.run(
            output_names=None,
            input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
        )

        labels, boxes, scores = output

        # Draw detections on the original frame
        result_images = draw(
            [frame_pil], labels, boxes, scores,
            [ratio], [(pad_w, pad_h)]
        )
        frame_with_detections = result_images[0]

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result.mp4'.")


def main(args):
    """Main function."""
    # Load the ONNX model

    sess = ort.InferenceSession(args.onnx)
    size = sess.get_inputs()[0].shape[2] #size 640
    
    #print(f"Using device: {ort.get_device()}")

    # 先导入必要的模块
    import os
    from pathlib import Path
    
    # 先将输入路径转换为Path对象并检查是否为目录
    input_path = Path(args.input)
    if input_path.is_dir():
        # 批量处理文件夹内所有图片
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        img_files = [p for p in input_path.iterdir() if p.suffix.lower() in img_exts]
        if not img_files:
            print("文件夹内未找到支持的图片文件！")
            return
        print(f"共发现 {len(img_files)} 张图片，开始批量推理...")
        for img_path in img_files:
            try:
                im_pil = Image.open(img_path).convert('RGB')
                start_time = time.time()
                # 传递原始图片路径作为保存路径
                process_image(sess, im_pil, str(img_path), size, args.model_size)
                end_time = time.time()
                print(f"{img_path.name} 推理完成，耗时: {end_time - start_time:.4f} 秒")
            except Exception as e:
                print(f"处理 {img_path.name} 时出错: {e}")
    else:
        # 单张图片或视频处理
        try:
            im_pil = Image.open(input_path).convert('RGB')
            #start_time = time.time()
            # 传递原始图片路径作为保存路径
            
            process_image(sess, im_pil, str(input_path), size, args.model_size)  # args.model_size ==s
            #end_time = time.time()
            #print(f"推理时间: {end_time - start_time:.4f} 秒")
        except IOError:
            # 非图片，按视频处理
            process_video(sess, input_path, size, args.model_size)
    # try:    
    #     # start_time = time.time()
    #     # process_image(sess, im_pil, size, args.model_size)
    #     # end_time = time.time()
    #     # print(f"推理时间: {end_time - start_time:.4f} 秒")
    # except IOError:
    #     # Not an image, process as video
    #     process_video(sess, input_path, size, args.model_size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or video file.')
    parser.add_argument('-ms', '--model-size', type=str, required=True, choices=['atto', 'femto', 'pico', 'n', 's', 'm', 'l', 'x'], 
                        help='Model size')
    args = parser.parse_args()
    main(args)
