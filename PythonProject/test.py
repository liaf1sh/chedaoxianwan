import cv2
import numpy as np


# 二次曲线拟合函数
def fit_curve(lane_points, y_values):
    if len(lane_points) == 0:
        return None
    # 提取x和y坐标
    x = np.array([point[0] for point in lane_points])
    y = np.array([point[1] for point in lane_points])

    # 二次曲线拟合
    coeffs = np.polyfit(y, x, 2)
    curve_fit = np.poly1d(coeffs)

    # 生成拟合曲线的x值
    curve_x = curve_fit(y_values)
    return curve_x


# 车道检测函数
def detect_lanes(frame):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blur, 50, 150)

    # 创建一个掩膜，只关注感兴趣区域
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)

    # 绘制直线车道
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # 检测弯道车道
    # 提取车道点
    lane_points = []
    for y in range(height // 2, height, 10):
        histogram = np.sum(masked_edges[y:y + 10, :], axis=0)
        x = np.argmax(histogram)
        if histogram[x] > 0:
            lane_points.append((x, y))

    # 使用二次曲线拟合弯道
    y_values = np.arange(height // 2, height, 10)
    curve_x = fit_curve(lane_points, y_values)

    # 绘制弯道车道
    if curve_x is not None:
        for i in range(len(y_values) - 1):
            cv2.line(frame, (int(curve_x[i]), y_values[i]), (int(curve_x[i + 1]), y_values[i + 1]), (0, 0, 255), 5)

    return frame


# 主函数
def main():
    # 读取视频
    cap = cv2.VideoCapture('video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 车道检测
        lane_frame = detect_lanes(frame)

        # 显示结果
        cv2.imshow('Lane Detection', lane_frame)

        # 按下 'q' 键退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        cv2.waitKey(10)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()