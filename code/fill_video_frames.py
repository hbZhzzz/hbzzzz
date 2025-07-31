import cv2
import numpy as np
import os

def convert_by_duplication(input_path, output_path, target_fps=30):
    """
    使用 OpenCV 通过复制帧的方式将视频帧率提升到目标值。
    """
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 {input_path}")
        return


    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return


    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频 '{os.path.basename(input_path)}' 的原始属性:")
    print(f"  帧率: {original_fps:.2f} FPS")
    print(f"  尺寸: {frame_width}x{frame_height}")
    print(f"  总帧数: {original_frame_count}")

    if original_fps >= target_fps:
        print("原始帧率已达到或超过目标帧率，无需转换。")
        cap.release()
        return
    print("正在读取所有原始帧到内存...")
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
    cap.release()
    print("读取完成。")


    new_frame_count = int(original_frame_count * (target_fps / original_fps))


    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者用 'X264'
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

    print(f"开始生成新视频，总帧数: {new_frame_count}")

    for i in range(new_frame_count):
        
        source_index = int(i * original_fps / target_fps)
        

        source_index = min(source_index, len(original_frames) - 1)
        
        frame_to_write = original_frames[source_index]
        writer.write(frame_to_write)


    writer.release()
    print(f"\n成功！转换后的视频已保存到: {output_path}")


if __name__ == '__main__':
    # video_dir = './stimuli/split_events_for_test/'
    # for video in os.listdir(video_dir):
    #     if video.endswith('mp4'):
    #         video_path = os.path.join(video_dir, video)
    #         output_video_path = os.path.join(video_dir, '30FPS', video)
    #         print(output_video_path)
    #         convert_by_duplication(video_path, output_video_path)
    
    pass