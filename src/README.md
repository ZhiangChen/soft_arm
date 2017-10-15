# Nodes
### 1. read_state
`bash
rosrun soft_arm read_state
` will run a node, 'read_state'. It receives the robot pose information from motion capture systems and target position, and then publishes [pose+target] as state to 'agent_state' topic. The message type is geometry_msgs/PoseArray.

### 2. naive_pose_pub.py
It publishes four poses information on one segment with the message type of geometry_msgs/PoseStamped. This node is for test purpose. The four topics are 'Robot_1/pose', 'Robot_2/pose', 'Robot_3/pose' and 'Robot_4/pose'. This node will be replaced by mocap_optitrack node in practice.

### 3. target_pub.py
It publishes the target position with the message type of geometry_msgs/PoseStamped to the topic 'Robot_5/pose'.

### 4. run_action.py
It publishes actions to low-level controller. It has three usages:
- `python run_action.py` publishes default air pressure {x:20,y:20,z:0}
- `python run_action.py 10 10 10` publishes three channels of air pressure
- `python run_action.py all` run entire actions in action space. And when the action is done, it will be published to a topic "action" with the message type of geometry_msgs/Vector3

### 5. action_state_recorder.py
It listens to 'agent_state' to get state information (robot pose + target position) and 'action' from run_action.py. It writes action-state pair to a file 'action_state_data'. See [the code](https://github.com/ZhiangChen/soft_arm/blob/master/src/action_state_recorder.py#L32) for more information about the data structure.
 
# Libraries
