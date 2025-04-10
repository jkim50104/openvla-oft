import threading
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraSubscriber(Node):
    def __init__(self, topic="/camera/camera/color/image_raw", node_name="camera_subscriber"):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.image = None
        self.lock = threading.Lock()

        self.sub = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.image = cv_img
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def get_image(self):
        with self.lock:
            image_copy = self.image.copy() if self.image is not None else None
            self.image = None  # clear buffer after fetching
        return image_copy