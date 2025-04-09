from rtde_receive import RTDEReceiveInterface

rtde_r = RTDEReceiveInterface("192.168.0.3")

# Get current TCP pose and joint angles
tcp_pose = rtde_r.getActualTCPPose()
joint_angles = rtde_r.getActualQ()

# Round and print like a real Python list
def format_list(lst):
    return "[" + ", ".join(f"{x:.3f}" for x in lst) + "]"

print("Current TCP pose:")
print(format_list(tcp_pose))

print("Current joint positions:")
print(format_list(joint_angles))